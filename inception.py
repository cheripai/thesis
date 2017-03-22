from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam


class Inception():
    
    model = None

    def __init__(self, outputs, lr, decay):
        inception_base = InceptionV3(weights="imagenet", include_top=False)
        self.model = self.add_top(inception_base, outputs)
        opt = Adam(lr=lr, decay=decay)
        self.model.compile(optimizer=opt, loss="mse")


    def add_top(self, base_model, outputs):
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(outputs)(x)
        model = Model(input=base_model.input, output=predictions)
        for layer in base_model.layers:
            layer.trainable = False
        return model
