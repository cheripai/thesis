from keras.applications.vgg16 import VGG16
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam


class VGG():
    
    model = None

    def __init__(self, outputs, lr, decay):
        vgg16_base = VGG16(weights="imagenet", include_top=False)
        self.model = self.add_top(vgg16_base, outputs)
        opt = Adam(lr=lr, decay=decay)
        self.model.compile(optimizer=opt, loss="mse")


    def add_top(self, base_model, outputs):
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.6)(x)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.6)(x)
        predictions = Dense(outputs)(x)
        model = Model(input=base_model.input, output=predictions)
        for layer in base_model.layers:
            layer.trainable = False
        return model
