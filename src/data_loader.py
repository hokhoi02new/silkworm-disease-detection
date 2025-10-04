import os
import tensorflow as tf
from config.config import IMG_HEIGHT, IMG_WIDTH 


def process_path(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH], method='nearest')
    mask = tf.cast(mask > 0, tf.float32)  

    return img, mask

def get_dataset(image_dir, mask_dir, batch_size=16, shuffle=True):
    images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
    
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100, reshuffle_each_iteration=True)
        
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
