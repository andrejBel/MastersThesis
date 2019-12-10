import constants

from global_functions import auto_str_repr, bind_method_to_instance
from datasets import Dataset

from tensorflow.keras import datasets, layers, models, losses, optimizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D, Input, BatchNormalization, \
    Dense, Dropout, AveragePooling2D
from tensorflow.keras.models import Model, Sequential

from abc import ABC, abstractmethod
from typing import Callable, Type


@auto_str_repr
class Models:

    def __init__(self, autoencoder: Model, classifier: Model, autoclassifier: Model):
        self.autoencoder = autoencoder
        self.classifier = classifier
        self.auto_classifier = autoclassifier
        self.compile_models()

    def compile_models(self, loss_weigh_decoder=1.0, loss_weight_classifier=2.0):
        self.autoencoder.compile(loss=losses.binary_crossentropy, optimizer=optimizers.RMSprop())
        self.classifier.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.Adam(),
                                metrics=['accuracy'])
        self.auto_classifier.compile(
            loss={constants.Models.DECODED_OUT: losses.binary_crossentropy,
                  constants.Models.CLASSIFIER_OUT: losses.categorical_crossentropy},
            loss_weights={constants.Models.DECODED_OUT: loss_weigh_decoder,
                          constants.Models.CLASSIFIER_OUT: loss_weight_classifier},
            optimizers={constants.Models.DECODED_OUT: optimizers.RMSprop(),
                        constants.Models.CLASSIFIER_OUT: optimizers.Adam()},
            metrics=["accuracy"]
        )

    def __iter__(self):
        return [self.autoencoder, self.classifier, self.auto_classifier].__iter__()

    def set_encoder_layers_trainable(self, trainable: bool):
        for layer in self.classifier.layers:
            layer.trainable = trainable
            if layer.name == constants.Models.ENCODED_OUT:
                break
        self.compile_models()


def make_classifier_with_softmax_activation(self):
    from tensorflow.keras import activations
    last_layer: Dense = self.classifier.layers[-1]
    last_layer.activation = activations.softmax
    self.compile_models()


def make_classifier_with_sigmoid_activation(self):
    from tensorflow.keras import activations
    last_layer: Dense = self.classifier.layers[-1]
    last_layer.activation = activations.sigmoid
    self.compile_models()


def make_classifier_without_activation(self):
    last_layer: Dense = self.classifier.layers[-1]
    last_layer.activation = None
    self.compile_models()


class ModelBuilder(ABC):

    def __init__(self, input_shape, number_of_classes: int):
        self.input_shape = input_shape
        self.number_of_classes = number_of_classes

    @abstractmethod
    def get_encoder(self, input_img: Input):
        pass

    @abstractmethod
    def get_decoder(self, encoded):
        pass

    @abstractmethod
    def get_fully_connected(self, enco: layers.Layer):
        pass

    def create_by_object(self):
        input_shape = Input(shape=self.input_shape, name=constants.Models.INPUT_SHAPE)
        encoder = Model(inputs=input_shape, outputs=self.get_encoder(input_shape), name='ENCODER')
        encoder_output_shape = Input(shape=encoder.output.shape[1:], name='DECODER_INPUT')
        decoder = Model(inputs=encoder_output_shape, outputs=self.get_decoder(encoder_output_shape), name='DECODER')
        classifier_head = Model(inputs=encoder_output_shape, outputs=self.get_fully_connected(encoder_output_shape),
                                name='CLASSIFIER_HEAD')
        autoencoder = Sequential([encoder, decoder], name=constants.Models.DECODED_OUT)
        classifier = Sequential([encoder, classifier_head], name=constants.Models.CLASSIFIER_OUT)
        auto_classifier = Model(inputs=input_shape, outputs=[autoencoder(input_shape), classifier(input_shape)],
                                name='auto_classifier')
        return Models(autoencoder, classifier, auto_classifier)

    def create_models(self) -> Models:
        input_shape = Input(shape=self.input_shape, name=constants.Models.INPUT_SHAPE)
        encoder = self.get_encoder(input_shape)
        decoder_output = self.get_decoder(encoder)
        autoencoder = Model(input_shape, decoder_output, name=constants.Models.AUTOENCODER)
        fully_connected_output = self.get_fully_connected(encoder)
        classifier = Model(input_shape, fully_connected_output, name=constants.Models.CLASSIFIER)
        auto_classifier = Model(inputs=input_shape, outputs=[autoencoder.outputs, fully_connected_output],
                                name=constants.Models.AUTO_CLASSIFIER)

        models = Models(autoencoder, classifier, auto_classifier)
        return models


class ModelProviderBase(ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Models:
        pass


class ModelProviderBuilder(ModelProviderBase):

    def __init__(self, clz: Type[ModelBuilder]) -> None:
        self.clz = clz

    def __call__(self, dataset: Dataset, **kwargs) -> Models:
        return self.clz(input_shape=dataset.get_input_shape(),
                        number_of_classes=dataset.get_number_of_classes()).create_models()


class BasicModelBuilder(ModelBuilder):

    def get_encoder(self, input_img: Input):
        # encoder
        # input = 28 x 28 x 1 (wide and thin)
        # e = BatchNormalization()(input_img)
        e = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  # 28 x 28 x 32
        e = BatchNormalization()(e)
        e = Conv2D(32, (3, 3), activation='relu', padding='same')(e)
        e = BatchNormalization()(e)
        e = MaxPooling2D(pool_size=(2, 2))(e)  # 14 x 14 x 32
        e = Dropout(0.1)(e)
        e = Conv2D(64, (3, 3), activation='relu', padding='same')(e)
        e = BatchNormalization()(e)
        e = MaxPooling2D(pool_size=(2, 2))(e)  # 7 x 7 x 64
        e = Dropout(0.1)(e)
        e = Conv2D(64, (3, 3), activation='relu', padding='same')(e)  # 7 x 7 x 64 (small and thick)
        e = BatchNormalization(name=constants.Models.ENCODED_OUT)(e)
        print('Nove OK')
        return e

    def get_decoder(self, encoded):
        # decoder
        d = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
        d = BatchNormalization()(d)
        d = UpSampling2D((2, 2))(d)  # 14 x 14 x 64
        d = Conv2D(64, (3, 3), activation='relu', padding='same')(d)
        d = BatchNormalization()(d)
        d = Conv2D(32, (3, 3), activation='relu', padding='same')(d)
        d = BatchNormalization()(d)
        d = UpSampling2D((2, 2))(d)  # 28 x 28 x 32
        d = Conv2D(32, (3, 3), activation='relu', padding='same')(d)
        d = BatchNormalization()(d)
        d = Conv2D(self.input_shape[2], (3, 3), activation='sigmoid',
                   padding='same',
                   name=constants.Models.DECODED_OUT)(d)
        return d

    def get_fully_connected(self, enco: layers.Layer):
        flat = Flatten()(enco)
        den = Dense(64, activation='relu')(flat)
        # den = Dense(64, activation='relu')(flat)
        den = Dropout(0.5)(den)
        # den = Dense(32, activation='relu')(den)
        # den = Dropout(0.4)(den)
        out = Dense(self.number_of_classes, activation='softmax', name=constants.Models.CLASSIFIER_OUT)(den)

        # convolution = Conv2D(64, (7, 7), activation='relu') (enco)
        # #den = Dense(64, activation='relu')(flat)
        # # den = Dense(64, activation='relu')(flat)
        # convolution = Dropout(0.5)(convolution)
        # flat = Flatten() (convolution)
        #
        # # den = Dense(32, activation='relu')(den)
        # # den = Dropout(0.4)(den)
        # out = Dense(self.number_of_classes, activation='softmax', name=constants.Models.CLASSIFIER_OUT)(flat)
        return out

    @staticmethod
    def get_provider() -> ModelProviderBase:
        return ModelProviderBuilder(BasicModelBuilder)


class BasicModelBuilderWithAveragePoolingWithDenseBuilder(BasicModelBuilder):

    def get_fully_connected(self, enco: layers.Layer):
        f = AveragePooling2D((7, 7))(enco)
        f = Flatten()(f)
        f = Dense(64, activation='relu')(f)
        f = Dropout(0.5)(f)
        out = Dense(self.number_of_classes, activation='softmax', name=constants.Models.CLASSIFIER_OUT)(f)
        return out

    @staticmethod
    def get_provider() -> ModelProviderBase:
        return ModelProviderBuilder(BasicModelBuilderWithAveragePoolingWithDenseBuilder)


class BasicModelBuilderWithAveragePoolingWithoutDenseBuilder(BasicModelBuilder):

    def get_fully_connected(self, enco: layers.Layer):
        f = AveragePooling2D((7, 7))(enco)
        f = Flatten()(f)
        f = Dropout(0.5)(f)
        out = Dense(self.number_of_classes, activation='softmax', name=constants.Models.CLASSIFIER_OUT)(f)
        return out

    @staticmethod
    def get_provider() -> ModelProviderBase:
        return ModelProviderBuilder(BasicModelBuilderWithAveragePoolingWithoutDenseBuilder)


class SmallModelBuilder(BasicModelBuilder):

    def get_encoder(self, input_img: Input):
        # encoder
        e = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        e = BatchNormalization()(e)
        e = MaxPooling2D(pool_size=(2, 2))(e)
        e = Conv2D(32, (3, 3), activation='relu', padding='same')(e)
        e = BatchNormalization()(e)
        e = MaxPooling2D(pool_size=(2, 2))(e)
        e = Dropout(0.1)(e)
        print('Nove OK')
        return e

    def get_decoder(self, encoded):
        # decoder
        d = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        d = BatchNormalization()(d)
        d = UpSampling2D((2, 2))(d)  # 14 x 14 x 64
        d = Conv2D(32, (3, 3), activation='relu', padding='same')(d)  # 7 x 7 x 64
        d = BatchNormalization()(d)
        d = UpSampling2D((2, 2))(d)  # 28 x 28 x 32
        d = Conv2D(self.input_shape[2], (3, 3), activation='sigmoid',
                   padding='same',
                   name=constants.Models.DECODED_OUT)(d)  # 28 x 28 x 1
        return d

    def get_fully_connected(self, enco: layers.Layer):
        f = AveragePooling2D((7, 7))(enco)
        f = Flatten()(f)
        # f = Dense(32, activation='relu')(f)
        # f = Dropout(0.5)(f)
        out = Dense(self.number_of_classes, activation='softmax', name=constants.Models.CLASSIFIER_OUT)(f)
        return out

    @staticmethod
    def get_provider() -> ModelProviderBase:
        return ModelProviderBuilder(SmallModelBuilder)


ModelProvider = Callable[..., Models]


class ExistingModelProvider(ModelProviderBase):

    def __init__(self, model_provider):
        if isinstance(model_provider, Models):
            self.model_provider = lambda: model_provider
        else:
            self.model_provider = model_provider

    def __call__(self, *args, **ignore) -> Models:
        return self.model_provider()
