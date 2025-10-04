import numpy as np
import tensorflow as tf
from data_loader import get_dataset
from config.config import TEST_IMG_DIR, TEST_MASK_DIR, BATCH_SIZE, SAVE_MODEL_DIR, SAVE_RESULT_DIR
import os 
import pandas as pd
import argparse

def calculate_metrics(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    true_negative = np.sum((y_true == 0) & (y_pred == 0))

    accuracy = np.mean(y_true == y_pred)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    dice_score = (2 * true_positive) / (2 * true_positive + false_positive + false_negative) if (2 * true_positive + false_positive + false_negative) > 0 else 0
    iou = true_positive / (true_positive + false_positive + false_negative) if (true_positive + false_positive + false_negative) > 0 else 0
    return accuracy, precision, recall, dice_score, iou

def show_metrics(model_name='unet_resnet', data_dir=TEST_IMG_DIR, mask_dir=TEST_MASK_DIR):
    test_ds = get_dataset(
        data_dir, 
        mask_dir, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )

    model_path = os.path.join(SAVE_MODEL_DIR, f"{model_name}.h5")
    model = tf.keras.models.load_model(model_path, compile=False)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    X_test = []
    y_test = []

    for x_batch, y_batch in test_ds:
        X_test.append(x_batch.numpy())
        y_test.append(y_batch.numpy())

    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)

    acc, prec, rec, dice, iou = calculate_metrics(y_test, y_pred)

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("Dice:", dice)
    print("IoU:", iou)

    result_path = os.path.join(SAVE_RESULT_DIR, f"{model_name}.csv")

    df = pd.DataFrame([{
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Dice": dice,
        "IoU": iou
    }])

    df.to_csv(result_path, index=False)
    print(f" Kết quả đã được lưu tại: {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument("--model", type=str, default="unet_resnet")
    parser.add_argument("--img_dir", type=str, default=TEST_IMG_DIR)
    parser.add_argument("--mask_dir", type=str, default=TEST_MASK_DIR)

    args = parser.parse_args()

    show_metrics(
        model_name=args.model,
        data_dir=args.img_dir,
        mask_dir=args.mask_dir
    )

    
    



