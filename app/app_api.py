from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import os
import uuid
import cv2
from ultralytics import YOLO
import shutil
from src.inference_video import inference_video_func
from src.inference_image import inference_image_func
import tensorflow as tf
from config.config import SAVE_MODEL_DIR, ALPHA, COLOR_MASK, UPLOAD_DIR, OUTPUT_DIR
import numpy as np

app = FastAPI()
unet_model = None
yolo_model = None

@app.on_event("startup")
async def startup_event():
    global unet_model, yolo_model
    try:
        print("loading model...")
        yolo_path = os.path.join(SAVE_MODEL_DIR, "YOLO.pt")
        unet_path = os.path.join(SAVE_MODEL_DIR, "unet_resnet.h5")

        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"YOLO model not found: {yolo_path}")
        if not os.path.exists(unet_path):
            raise FileNotFoundError(f"UNet model not found: {unet_path}")

        yolo_model = YOLO(yolo_path)
        unet_model = tf.keras.models.load_model(unet_path, compile=False)

        print("model loaded successful")
    except Exception as e:
        print(f"failed to load models: {e}")
        raise HTTPException(status_code=500, detail=f"failed to load model: {str(e)}")
    
@app.get("/")
async def root():
    return {"message": "Silkworm disease API is running"}

@app.get("/health")
async def health_check():   
    if yolo_model is None or unet_model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model_loaded": True, "message": "API is running healthy"}  

@app.post("/predict_video/")
async def inference_video_api(file: UploadFile = File(...)):
    input_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    output_path = os.path.join(OUTPUT_DIR, f"output_{file.filename}")

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        inference_video_func(video_path=input_path, model_inference=yolo_model, output_path=output_path, show=False)

        return FileResponse(output_path, media_type="video/mp4", filename="result.mp4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"video inference failed: {str(e)}")

@app.post("/predict_image/")
async def inference_image_api(file: UploadFile = File(...)):
    input_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    output_path = os.path.join(OUTPUT_DIR, f"output_{file.filename}")

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        orig_img, mask = inference_image_func(unet_model, input_path)

        overlay = orig_img.copy().astype(np.float32)

        color_layer = np.full_like(orig_img, COLOR_MASK, dtype=np.uint8)
        color_layer = color_layer.astype(np.float32) / 255.0  

        mask_3d = np.repeat(mask[..., None], 3, axis=-1).astype(bool)

        overlay[mask_3d == 1] = (1 - ALPHA) * overlay[mask_3d == 1] + ALPHA * color_layer[mask_3d == 1]


        overlay_bgr = (overlay * 255).astype("uint8")
        overlay_bgr = cv2.cvtColor(overlay_bgr, cv2.COLOR_RGB2BGR)

        
        cv2.imwrite(output_path, overlay_bgr)

        return FileResponse(output_path, media_type="image/jpeg", filename="result.jpg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"image inference failed: {str(e)}")