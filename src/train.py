import tensorflow as tf
import argparse
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
from src.data_loader import get_dataset
from utils.utils import bce_dice_loss_combine
from src.models.unet_resnet import UNetResNet50
from src.models.deeplabv3_plus import DeepLabV3Plus
from segmentation_models.metrics import IOUScore, FScore
from config.config import BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, SAVE_MODEL_DIR, EPOCHS, TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, IMG_CHANNELS

def train(model_name="unet_resnet", epochs=EPOCHS, batch_size=BATCH_SIZE, save_dir=SAVE_MODEL_DIR):
    train_ds = get_dataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, batch_size=batch_size, shuffle=True)
    val_ds   = get_dataset(VAL_IMG_DIR, VAL_MASK_DIR, batch_size=batch_size, shuffle=False)

    if model_name.lower() == "unet_resnet":
        model = UNetResNet50(input_size=(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS), backbone_trainable=True).get_model()
    elif model_name.lower() == "deeplabv3+":
        model = DeepLabV3Plus(input_size=(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS), backbone_trainable=True).get_model()
    else:
        raise ValueError("model does not support choose unet_resnet or deeplabv3+ ")

    model.compile(
        optimizer="adam",
        loss=bce_dice_loss_combine,
        metrics=['accuracy', IOUScore(threshold=0.5), FScore(threshold=0.5)]
    )

    

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        )]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    save_path = os.path.join(save_dir, f"{model_name}.h5")
    print(f"training done, model save as {save_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--model_name", type=str, default="unet_resnet")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=SAVE_MODEL_DIR
    )