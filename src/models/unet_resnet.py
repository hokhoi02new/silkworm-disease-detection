import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from config.config import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS

class UNetResNet50:
    def __init__(self, input_size=(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),backbone_trainable=True):
        self.input_size = input_size
        self.backbone_trainable = backbone_trainable
        self.model = self.build_model()
    
    def build_model(self):
        #encoder (using resnet50 pretrained)
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=self.input_size)

        c1 = base_model.get_layer("conv1_relu").output          
        c2 = base_model.get_layer("conv2_block3_out").output    
        c3 = base_model.get_layer("conv3_block4_out").output    
        c4 = base_model.get_layer("conv4_block6_out").output    
        c5 = base_model.get_layer("conv5_block3_out").output    

        if self.backbone_trainable == False:
            for layer in base_model.layers:
                layer.trainable = False

        #decoder
        u6 = layers.Conv2DTranspose(1024, (2,2), strides=2, padding="same")(c5) 
        u6 = layers.concatenate([u6, c4])
        c6 = layers.Conv2D(1024, 3, activation="relu", padding="same")(u6)
        c6 = layers.Conv2D(1024, 3, activation="relu", padding="same")(c6)

        u7 = layers.Conv2DTranspose(512, (2,2), strides=2, padding="same")(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = layers.Conv2D(512, 3, activation="relu", padding="same")(u7)
        c7 = layers.Conv2D(512, 3, activation="relu", padding="same")(c7)

        u8 = layers.Conv2DTranspose(256, (2,2), strides=2, padding="same")(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = layers.Conv2D(256, 3, activation="relu", padding="same")(u8)
        c8 = layers.Conv2D(256, 3, activation="relu", padding="same")(c8)

        u9 = layers.Conv2DTranspose(128, (2,2), strides=2, padding="same")(c8)
        u9 = layers.concatenate([u9, c1])
        c9 = layers.Conv2D(128, 3, activation="relu", padding="same")(u9)
        c9 = layers.Conv2D(128, 3, activation="relu", padding="same")(c9)

        u10 = layers.Conv2DTranspose(64, (2,2), strides=2, padding="same")(c9)
        c10 = layers.Conv2D(64, 3, activation="relu", padding="same")(u10)
        c10 = layers.Conv2D(64, 3, activation="relu", padding="same")(c10)

        outputs = layers.Conv2D(1, (1,1), activation="sigmoid")(c10)

        model = models.Model(inputs=base_model.input, outputs=outputs)
        return model


    def get_model(self):
        return self.model
