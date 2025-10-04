import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
def bce_dice_loss_combine(y_true, y_pred, alpha=0.6):
    bce = BinaryCrossentropy()(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    return (alpha * bce) + (1 - alpha) * d_loss
