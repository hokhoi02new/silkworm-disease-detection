import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from config.config import IMG_HEIGHT, IMG_WIDTH, THRESHOLD, ALPHA, COLOR_MASK, SAVE_MODEL_DIR
import argparse
import os

def inference_image_func(model, img_path):
    img_size = (IMG_HEIGHT,IMG_WIDTH)
    threshold = THRESHOLD

    orig_img = cv2.imread(img_path)

    if orig_img is None:
        raise FileNotFoundError(f"not found image: {img_path}")

    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    h, w, _ = orig_img.shape

    resized_img = cv2.resize(orig_img, img_size, interpolation=cv2.INTER_LINEAR)
    img_array = resized_img.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    predicted_mask = model.predict(img_input)[0]
    binary_mask = (predicted_mask > threshold).astype(np.uint8)[:,:,0]

    mask_resized = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return orig_img.astype(np.float32)/255.0, mask_resized

def visualize_overlay(orig_img, mask):
    alpha = ALPHA
    color = COLOR_MASK
    overlay = orig_img.copy()
    color_layer = np.zeros_like(orig_img)
    color_layer[:,:] = color
    overlay = np.where(mask[...,None]==1,
                       (1-alpha)*orig_img + alpha*color_layer/255.0,
                       orig_img)

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title("Original image")
    plt.imshow(orig_img)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Overlay Mask")
    plt.imshow(overlay)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference segmentation model")
    parser.add_argument("--model", type=str,  default="unet_resnet")
    parser.add_argument("--image_path", type=str, default='data/test_sample/image_3964.jpg')

    args = parser.parse_args()

    model_name = args.model
    model_path = os.path.join(SAVE_MODEL_DIR, f"{model_name}.h5")

    model = tf.keras.models.load_model(model_path, compile=False)

    orig_img, mask = inference_image_func(model, args.image_path)

    visualize_overlay(orig_img, mask)
