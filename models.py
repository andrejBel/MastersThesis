from constants import Constants

from global_functions import bind_method_to_instance

from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D, Input, BatchNormalization, \
    Dense, Dropout
from tensorflow.keras.models import Model, Sequential


class Models:

    def __init__(self, autoencoder: Model, classifier: Model, autoclassifier: Model):
        self.autoencoder = autoencoder
        self.classifier = classifier
        self.auto_classifier = autoclassifier


class BasicModelBuilder:

    def __init__(self, input_shape, number_of_classes: int):
        self.activation = 'relu'
        self.input_shape = input_shape
        self.number_of_classes = number_of_classes

    def get_encoder(self, input_img: Input):
        # encoder
        # input = 28 x 28 x 1 (wide and thin)
        e = Conv2D(32, (3, 3), activation=self.activation, padding='same')(input_img)  # 28 x 28 x 32
        e = BatchNormalization()(e)
        e = Conv2D(32, (3, 3), activation=self.activation, padding='same')(e)
        e = BatchNormalization()(e)
        e = MaxPooling2D(pool_size=(2, 2))(e)  # 14 x 14 x 32
        # e = Conv2D(64, (3, 3), activation=activation, padding='same')(e) #14 x 14 x 64
        # e = BatchNormalization()(e)
        e = Conv2D(64, (3, 3), activation=self.activation, padding='same')(e)
        e = BatchNormalization()(e)
        e = MaxPooling2D(pool_size=(2, 2))(e)  # 7 x 7 x 64
        e = Conv2D(128, (3, 3), activation=self.activation, padding='same')(e)  # 7 x 7 x 128 (small and thick)
        e = BatchNormalization(name=Constants.Models.ENCODED_OUT)(e)
        return e

    def get_decoder(self, encoded):
        # decoder
        d = Conv2D(128, (3, 3), activation=self.activation, padding='same')(encoded)
        d = BatchNormalization()(d)
        d = Conv2D(64, (3, 3), activation=self.activation, padding='same')(d)  # 7 x 7 x 64
        d = BatchNormalization()(d)
        d = Conv2D(64, (3, 3), activation=self.activation, padding='same')(d)
        d = BatchNormalization()(d)
        d = UpSampling2D((2, 2))(d)  # 14 x 14 x 64
        # d = Conv2D(32, (3, 3), activation=activation, padding='same')(d) # 14 x 14 x 32
        # d = BatchNormalization()(d)
        d = Conv2D(32, (3, 3), activation=self.activation, padding='same')(d)
        d = BatchNormalization()(d)
        d = UpSampling2D((2, 2))(d)  # 28 x 28 x 32
        d = Conv2D(self.input_shape[2], (3, 3), activation='sigmoid',
                   padding='same',
                   name=Constants.Models.DECODED_OUT)(d)  # 28 x 28 x 1
        return d

    def get_fully_connected(self, enco: layers.Layer):
        flat = Flatten()(enco)
        den = Dense(128, activation='relu')(flat)
        den = Dropout(0.3)(den)
        den = Dense(32, activation='relu')(den)
        den = Dropout(0.4)(den)
        out = Dense(self.number_of_classes, activation='softmax', name=Constants.Models.CLASSIFIER_OUT)(den)
        return out

    def create_by_object(self):
        input_shape = Input(shape=self.input_shape, name=Constants.Models.INPUT_SHAPE)
        encoder = Model(inputs=input_shape, outputs=self.get_encoder(input_shape), name='ENCODER')
        encoder_output_shape = Input(shape=encoder.output.shape[1:], name='DECODER_INPUT')
        decoder = Model(inputs=encoder_output_shape, outputs=self.get_decoder(encoder_output_shape), name='DECODER')
        classifier_head = Model(inputs=encoder_output_shape, outputs=self.get_fully_connected(encoder_output_shape),
                                name='CLASSIFIER_HEAD')
        autoencoder = Sequential([encoder, decoder], name=Constants.Models.DECODED_OUT)
        classifier = Sequential([encoder, classifier_head], name=Constants.Models.CLASSIFIER_OUT)
        auto_classifier = Model(inputs=input_shape, outputs=[autoencoder(input_shape), classifier(input_shape)],
                                name='auto_classifier')
        return Models(autoencoder, classifier, auto_classifier)

    def compile_models(self, models: Models):
        models.autoencoder.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.RMSprop())
        models.classifier.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                                  metrics=['accuracy'])
        models.auto_classifier.compile(
            loss={Constants.Models.DECODED_OUT: keras.losses.binary_crossentropy,
                  Constants.Models.CLASSIFIER_OUT: keras.losses.categorical_crossentropy},
            loss_weights={Constants.Models.DECODED_OUT: 1.0, Constants.Models.CLASSIFIER_OUT: 2.0},
            optimizers={Constants.Models.DECODED_OUT: keras.optimizers.RMSprop(),
                        Constants.Models.CLASSIFIER_OUT: keras.optimizers.RMSprop()},
            metrics=["accuracy"]
        )

    def add_methods_to_models(self, models: Models):
        def set_autoencoder_trainable(selfinner, innervalue):
            for layer in selfinner.layers:
                layer.trainable = innervalue

        models.autoencoder.set_autoencoder_trainable = bind_method_to_instance(models.autoencoder,
                                                                               set_autoencoder_trainable)

        def set_autoencoder_layers_trainable(selfinner, innervalue):
            for layer in selfinner.layers:
                layer.trainable = innervalue
                if layer.name == Constants.Models.ENCODED_OUT:
                    break

        models.classifier.set_autoencoder_layers_trainable = bind_method_to_instance(models.classifier,
                                                                                     set_autoencoder_layers_trainable)

    def create_models(self) -> Models:
        input_shape = Input(shape=self.input_shape, name=Constants.Models.INPUT_SHAPE)
        encoder = self.get_encoder(input_shape)
        decoder_output = self.get_decoder(encoder)
        autoencoder = Model(input_shape, decoder_output)
        fully_connected_output = self.get_fully_connected(encoder)
        classifier = Model(input_shape, fully_connected_output)
        auto_classifier = Model(inputs=input_shape, outputs=[autoencoder.outputs, fully_connected_output])

        models = Models(autoencoder, classifier, auto_classifier)
        self.compile_models(models)
        self.add_methods_to_models(models)
        return models


import datasets

models = BasicModelBuilder(datasets.MnistDataset().get_input_shape(),
                           datasets.MnistDataset().get_number_of_classes()).create_models()
models.classifier.set_autoencoder_layers_trainable(False)
