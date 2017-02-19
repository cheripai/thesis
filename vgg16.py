import numpy as np
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential


class VGG16():

    weights_path = "data/vgg16_bn.h5"
    model = None
    last_conv_idx = None

    def __init__(self, dropout=0.5):
        self.build(dropout)

    def vgg_preprocess(self, x):
        vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((3, 1, 1))
        x = x - vgg_mean
        return x[:, ::-1]

    def conv_block(self, layers, filters):
        for i in range(layers):
            self.model.add(ZeroPadding2D((1, 1)))
            self.model.add(Convolution2D(filters, 3, 3, activation="relu"))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    def fc_block(self, dropout):
        self.model.add(Dense(4096, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout))

    def build(self, dropout):
        self.model = Sequential()
        self.model.add(Lambda(self.vgg_preprocess, input_shape=(3, 224, 224)))

        self.conv_block(2, 64)
        self.conv_block(2, 128)
        self.conv_block(3, 256)
        self.conv_block(3, 512)
        self.conv_block(3, 512)
        self.model.add(Flatten())

        self.fc_block(dropout)
        self.fc_block(dropout)
        self.model.add(Dense(1000, activation="softmax"))
        self.model.load_weights(self.weights_path)

        self.last_conv_idx = [idx for idx, layer in enumerate(self.model.layers) if type(layer) is Convolution2D][-1]

    def finetune_last(self, num_outputs):
        self.model.pop()
        for layer in self.model.layers:
            layer.trainable = False
        self.model.add(Dense(num_outputs, activation="softmax"))

    def finetune_dense(self, num_outputs):
        self.model.pop()
        for layer in self.model.layers[:self.last_conv_idx+1]:
            layer.trainable = False
        self.model.add(Dense(num_outputs, activation="softmax"))

    def scale_weights(self, layer, prev_p, new_p):
        scal = (1 - prev_p) / (1 - new_p)
        return [o * scal for o in layer.get_weights()]


if __name__ == "__main__":
    vgg16 = VGG16()
    print(vgg16.model.summary())
