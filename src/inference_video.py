import cv2
import numpy as np
import argparse
from ultralytics import YOLO


def inference_video_func(video_path, model_inference, output_path="data/test_sample/video_output.mp4", show=False):
    model = model_inference
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"can't open video: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 25.0 
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)

        if results[0].masks is not None:
            overlay = frame.copy()
            for m in results[0].masks.xy:
                pts = m.astype(np.int32)
                cv2.fillPoly(overlay, [pts], (0, 0, 255))
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for (box, conf, cls) in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                label = model.names.get(cls, str(cls))
                text = f"{label} {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        out.write(frame)
        if show == True:
            cv2.imshow("YOLOv8-Seg Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"video output save as: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8-Seg Video Inference")
    parser.add_argument("--video", type=str,  default="data/test_sample/test_video.mp4")
    parser.add_argument("--output", type=str, default="data/test_sample/video_output.mp4")
    model_path = 'save_models/YOLO.pt'
    model = YOLO(model_path)
    args = parser.parse_args()
    inference_video_func(video_path=args.video, model_inference=model, output_path=args.output, show=True)
