from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam


class CNN:

    model = None

    def __init__(self, outputs, lr, decay, dense, dropout, classification):

        if type(self).__name__ == "VGG":
            base = VGG16(weights="imagenet", include_top=False)
        elif type(self).__name__ == "ResNet":
            base = ResNet50(weights="imagenet", include_top=False)
        elif type(self).__name__ == "Inception":
            base = InceptionV3(weights="imagenet", include_top=False)
        else:
            raise ValueError("Invalid model name: {}".format(type(self).__name__))
        self.classification = classification

        self.model = self.add_top(base, outputs, dense, dropout)
        opt = Adam(lr=lr, decay=decay)
        if self.classification == True:
            self.model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["acc"])
        else:
            self.model.compile(optimizer=opt, loss="mse")


    def add_top(self, base_model, outputs, dense, dropout):
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(dense, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Dense(dense, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        if self.classification == True:
            predictions = Dense(outputs, activation="softmax")(x)
        else:
            predictions = Dense(outputs)(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        for layer in base_model.layers:
            layer.trainable = False
        return model


class VGG(CNN):
    def __init__(self, outputs, lr, decay, dense=512, dropout=0.6, classification=False):
        super(self.__class__, self).__init__(outputs, lr, decay, dense, dropout, classification)


class Inception(CNN):
    def __init__(self, outputs, lr, decay, dense=512, dropout=0.5, classification=False):
        super(self.__class__, self).__init__(outputs, lr, decay, dense, dropout, classification)


class ResNet(CNN):
    def __init__(self, outputs, lr, decay, dense=512, dropout=0.5, classification=False):
        super(self.__class__, self).__init__(outputs, lr, decay, dense, dropout, classification)
