import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from config.config import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS

def ASPP(x, filters): #ASPP (Atrous Spatial Pyramid Pooling)
    shape = x.shape

    y1 = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    y1 = layers.BatchNormalization()(y1)
    y1 = layers.ReLU()(y1)

    y2 = layers.Conv2D(filters, 3, dilation_rate=6, padding="same", use_bias=False)(x)
    y2 = layers.BatchNormalization()(y2)
    y2 = layers.ReLU()(y2)

    y3 = layers.Conv2D(filters, 3, dilation_rate=12, padding="same", use_bias=False)(x)
    y3 = layers.BatchNormalization()(y3)
    y3 = layers.ReLU()(y3)

    y4 = layers.Conv2D(filters, 3, dilation_rate=18, padding="same", use_bias=False)(x)
    y4 = layers.BatchNormalization()(y4)
    y4 = layers.ReLU()(y4)

    #global average pooling branch
    y5 = layers.GlobalAveragePooling2D()(x)
    y5 = layers.Reshape((1,1,shape[-1]))(y5)
    y5 = layers.Conv2D(filters, 1, padding="same", use_bias=False)(y5)
    y5 = layers.BatchNormalization()(y5)
    y5 = layers.ReLU()(y5)
    y5 = layers.UpSampling2D(size=(shape[1], shape[2]), interpolation="bilinear")(y5)

    y = layers.Concatenate()([y1, y2, y3, y4, y5])
    y = layers.Conv2D(filters, 1, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    return y

class DeepLabV3Plus:
    def __init__(self, input_size=(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS), backbone_trainable=True):
        self.input_size = input_size
        self.backbone_trainable = backbone_trainable
        self.model = self.build_model()

    def build_model(self):
        #encoder (using ResNet50 pretrained)
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=self.input_size)
        
        if self.backbone_trainable == False:
            for layer in base_model.layers:
                layer.trainable = False

        #feature map
        high_level = base_model.get_layer("conv4_block6_out").output  # stride 16
        low_level  = base_model.get_layer("conv2_block3_out").output  # stride 4

        #ASPP module
        x = ASPP(high_level, 256)
        x = layers.UpSampling2D(size=(4,4), interpolation="bilinear")(x)  # stride 16 → stride 4

        #decoder
        low_level = layers.Conv2D(48, 1, padding="same", use_bias=False)(low_level)
        low_level = layers.BatchNormalization()(low_level)
        low_level = layers.ReLU()(low_level)

        x = layers.Concatenate()([x, low_level])
        x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        #upsampling
        x = layers.UpSampling2D(size=(4,4), interpolation="bilinear")(x)
        outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)

        return models.Model(inputs=base_model.input, outputs=outputs)

    def get_model(self):
        return self.model
