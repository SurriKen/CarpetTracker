import importlib
import json
import os

import numpy as np
import tensorflow as tf
import tensorflow
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Layer


class PretrainedYOLO(Layer):

    def __init__(self, num_classes: int = 5, version: str = "YOLOv4",
                 use_weights: bool = True, save_weights: str = '', **kwargs):
        super(PretrainedYOLO, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.version = version
        self.use_weights = use_weights
        self.save_weights = str(save_weights)
        self.yolo = self.create_yolo(classes=self.num_classes)
        if use_weights:
            self.base_yolo = self.create_yolo()
            self.load_yolo_weights(self.base_yolo, self.save_weights)
            for i, l in enumerate(self.base_yolo.layers):
                layer_weights = l.get_weights()
                if layer_weights:
                    try:
                        self.yolo.layers[i].set_weights(layer_weights)
                    except:
                        print("skipping", self.yolo.layers[i].name)
            del self.base_yolo

    def create_yolo(self, input_size=416, channels=3, classes=80):
        tf.keras.backend.clear_session()  # used to reset layer names
        input_layer = layers.Input([input_size, input_size, channels])
        if self.version == "YOLOv4":
            output_tensors = self.YOLOv4(input_layer, classes)
        else:
            output_tensors = self.YOLOv3(input_layer, classes)
        yolo = tf.keras.Model(input_layer, output_tensors)
        return yolo

    def load_yolo_weights(self, model, weights_file):
        # print(weights_file)
        tf.keras.backend.clear_session()  # used to reset layer names
        # load Darknet original weights to TensorFlow model
        if self.version == "YOLOv3":
            range1 = 75
            range2 = [58, 66, 74]
        if self.version == "YOLOv4":
            range1 = 110
            range2 = [93, 101, 109]

        with open(weights_file, 'rb') as wf:
            major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

            j = 0
            for i in range(range1):
                if i > 0:
                    conv_layer_name = 'conv2d_%d' % i
                else:
                    conv_layer_name = 'conv2d'

                if j > 0:
                    bn_layer_name = 'batch_normalization_%d' % j
                else:
                    bn_layer_name = 'batch_normalization'

                conv_layer = model.get_layer(conv_layer_name)
                filters = conv_layer.filters
                k_size = conv_layer.kernel_size[0]
                in_dim = conv_layer.input_shape[-1]

                if i not in range2:
                    # darknet weights: [beta, gamma, mean, variance]
                    bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                    # tf weights: [gamma, beta, mean, variance]
                    bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                    bn_layer = model.get_layer(bn_layer_name)
                    j += 1
                else:
                    conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

                # darknet shape (out_dim, in_dim, height, width)
                conv_shape = (filters, in_dim, k_size, k_size)
                conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
                # tf shape (height, width, in_dim, out_dim)
                conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

                if i not in range2:
                    conv_layer.set_weights([conv_weights])
                    bn_layer.set_weights(bn_weights)
                else:
                    conv_layer.set_weights([conv_weights, conv_bias])

            assert len(wf.read()) == 0, 'failed to read all data'

    def convolutional(self, input_layer, filters_shape, downsample=False, activate=True, bn=True,
                      activate_type='leaky'):
        if downsample:
            input_layer = layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
            padding = 'valid'
            strides = 2
        else:
            strides = 1
            padding = 'same'

        conv = layers.Conv2D(filters=filters_shape[-1], kernel_size=filters_shape[0], strides=strides,
                             padding=padding, use_bias=not bn,
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                             kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                             bias_initializer=tf.constant_initializer(0.))(input_layer)
        if bn:
            conv = BatchNormalization()(conv)
        if activate:
            if activate_type == "leaky":
                conv = layers.LeakyReLU(alpha=0.1)(conv)
            elif activate_type == "mish":
                conv = self.mish(conv)
        return conv

    def mish(self, x):
        return x * tf.math.tanh(tf.math.softplus(x))

    def residual_block(self, input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
        short_cut = input_layer
        conv = self.convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1),
                                  activate_type=activate_type)
        conv = self.convolutional(conv, filters_shape=(3, 3, filter_num1, filter_num2), activate_type=activate_type)

        residual_output = short_cut + conv
        return residual_output

    def upsample(self, input_layer):
        return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')

    def route_group(self, input_layer, groups, group_id):
        convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
        return convs[group_id]

    def darknet53(self, input_data):
        input_data = self.convolutional(input_data, (3, 3, 3, 32))
        input_data = self.convolutional(input_data, (3, 3, 32, 64), downsample=True)

        for i in range(1):
            input_data = self.residual_block(input_data, 64, 32, 64)

        input_data = self.convolutional(input_data, (3, 3, 64, 128), downsample=True)

        for i in range(2):
            input_data = self.residual_block(input_data, 128, 64, 128)

        input_data = self.convolutional(input_data, (3, 3, 128, 256), downsample=True)

        for i in range(8):
            input_data = self.residual_block(input_data, 256, 128, 256)

        route_1 = input_data
        input_data = self.convolutional(input_data, (3, 3, 256, 512), downsample=True)

        for i in range(8):
            input_data = self.residual_block(input_data, 512, 256, 512)

        route_2 = input_data
        input_data = self.convolutional(input_data, (3, 3, 512, 1024), downsample=True)

        for i in range(4):
            input_data = self.residual_block(input_data, 1024, 512, 1024)

        return route_1, route_2, input_data

    def cspdarknet53(self, input_data):
        input_data = self.convolutional(input_data, (3, 3, 3, 32), activate_type="mish")
        input_data = self.convolutional(input_data, (3, 3, 32, 64), downsample=True, activate_type="mish")

        route = input_data
        route = self.convolutional(route, (1, 1, 64, 64), activate_type="mish")
        input_data = self.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
        for i in range(1):
            input_data = self.residual_block(input_data, 64, 32, 64, activate_type="mish")
        input_data = self.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")

        input_data = tf.concat([input_data, route], axis=-1)
        input_data = self.convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
        input_data = self.convolutional(input_data, (3, 3, 64, 128), downsample=True, activate_type="mish")
        route = input_data
        route = self.convolutional(route, (1, 1, 128, 64), activate_type="mish")
        input_data = self.convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
        for i in range(2):
            input_data = self.residual_block(input_data, 64, 64, 64, activate_type="mish")
        input_data = self.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
        input_data = tf.concat([input_data, route], axis=-1)

        input_data = self.convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
        input_data = self.convolutional(input_data, (3, 3, 128, 256), downsample=True, activate_type="mish")
        route = input_data
        route = self.convolutional(route, (1, 1, 256, 128), activate_type="mish")
        input_data = self.convolutional(input_data, (1, 1, 256, 128), activate_type="mish")
        for i in range(8):
            input_data = self.residual_block(input_data, 128, 128, 128, activate_type="mish")
        input_data = self.convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
        input_data = tf.concat([input_data, route], axis=-1)

        input_data = self.convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
        route_1 = input_data
        input_data = self.convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type="mish")
        route = input_data
        route = self.convolutional(route, (1, 1, 512, 256), activate_type="mish")
        input_data = self.convolutional(input_data, (1, 1, 512, 256), activate_type="mish")
        for i in range(8):
            input_data = self.residual_block(input_data, 256, 256, 256, activate_type="mish")
        input_data = self.convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
        input_data = tf.concat([input_data, route], axis=-1)

        input_data = self.convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
        route_2 = input_data
        input_data = self.convolutional(input_data, (3, 3, 512, 1024), downsample=True, activate_type="mish")
        route = input_data
        route = self.convolutional(route, (1, 1, 1024, 512), activate_type="mish")
        input_data = self.convolutional(input_data, (1, 1, 1024, 512), activate_type="mish")
        for i in range(4):
            input_data = self.residual_block(input_data, 512, 512, 512, activate_type="mish")
        input_data = self.convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
        input_data = tf.concat([input_data, route], axis=-1)

        input_data = self.convolutional(input_data, (1, 1, 1024, 1024), activate_type="mish")
        input_data = self.convolutional(input_data, (1, 1, 1024, 512))
        input_data = self.convolutional(input_data, (3, 3, 512, 1024))
        input_data = self.convolutional(input_data, (1, 1, 1024, 512))

        max_pooling_1 = tf.keras.layers.MaxPool2D(pool_size=13, padding='SAME', strides=1)(input_data)
        max_pooling_2 = tf.keras.layers.MaxPool2D(pool_size=9, padding='SAME', strides=1)(input_data)
        max_pooling_3 = tf.keras.layers.MaxPool2D(pool_size=5, padding='SAME', strides=1)(input_data)
        input_data = tf.concat([max_pooling_1, max_pooling_2, max_pooling_3, input_data], axis=-1)

        input_data = self.convolutional(input_data, (1, 1, 2048, 512))
        input_data = self.convolutional(input_data, (3, 3, 512, 1024))
        input_data = self.convolutional(input_data, (1, 1, 1024, 512))

        return route_1, route_2, input_data

    def YOLOv3(self, input_layer, NUM_CLASS):
        # After the input layer enters the Darknet-53 network, we get three branches
        route_1, route_2, conv = self.darknet53(input_layer)
        # See the orange module (DBL) in the figure above, a total of 5 Subconvolution operation
        conv = self.convolutional(conv, (1, 1, 1024, 512))
        conv = self.convolutional(conv, (3, 3, 512, 1024))
        conv = self.convolutional(conv, (1, 1, 1024, 512))
        conv = self.convolutional(conv, (3, 3, 512, 1024))
        conv = self.convolutional(conv, (1, 1, 1024, 512))
        conv_lobj_branch = self.convolutional(conv, (3, 3, 512, 1024))

        # conv_lbbox is used to predict large-sized objects , Shape = [None, 13, 13, 255]
        conv_lbbox = self.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        conv = self.convolutional(conv, (1, 1, 512, 256))
        # upsample here uses the nearest neighbor interpolation method, which has the advantage that the
        # upsampling process does not need to learn, thereby reducing the network parameter
        conv = self.upsample(conv)

        conv = tf.concat([conv, route_2], axis=-1)
        conv = self.convolutional(conv, (1, 1, 768, 256))
        conv = self.convolutional(conv, (3, 3, 256, 512))
        conv = self.convolutional(conv, (1, 1, 512, 256))
        conv = self.convolutional(conv, (3, 3, 256, 512))
        conv = self.convolutional(conv, (1, 1, 512, 256))
        conv_mobj_branch = self.convolutional(conv, (3, 3, 256, 512))

        # conv_mbbox is used to predict medium-sized objects, shape = [None, 26, 26, 255]
        conv_mbbox = self.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        conv = self.convolutional(conv, (1, 1, 256, 128))
        conv = self.upsample(conv)

        conv = tf.concat([conv, route_1], axis=-1)
        conv = self.convolutional(conv, (1, 1, 384, 128))
        conv = self.convolutional(conv, (3, 3, 128, 256))
        conv = self.convolutional(conv, (1, 1, 256, 128))
        conv = self.convolutional(conv, (3, 3, 128, 256))
        conv = self.convolutional(conv, (1, 1, 256, 128))
        conv_sobj_branch = self.convolutional(conv, (3, 3, 128, 256))

        # conv_sbbox is used to predict small size objects, shape = [None, 52, 52, 255]
        conv_sbbox = self.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        return [conv_sbbox, conv_mbbox, conv_lbbox]

    def YOLOv4(self, input_layer, NUM_CLASS):
        route_1, route_2, conv = self.cspdarknet53(input_layer)

        route = conv
        conv = self.convolutional(conv, (1, 1, 512, 256))
        conv = self.upsample(conv)
        route_2 = self.convolutional(route_2, (1, 1, 512, 256))
        conv = tf.concat([route_2, conv], axis=-1)

        conv = self.convolutional(conv, (1, 1, 512, 256))
        conv = self.convolutional(conv, (3, 3, 256, 512))
        conv = self.convolutional(conv, (1, 1, 512, 256))
        conv = self.convolutional(conv, (3, 3, 256, 512))
        conv = self.convolutional(conv, (1, 1, 512, 256))

        route_2 = conv
        conv = self.convolutional(conv, (1, 1, 256, 128))
        conv = self.upsample(conv)
        route_1 = self.convolutional(route_1, (1, 1, 256, 128))
        conv = tf.concat([route_1, conv], axis=-1)

        conv = self.convolutional(conv, (1, 1, 256, 128))
        conv = self.convolutional(conv, (3, 3, 128, 256))
        conv = self.convolutional(conv, (1, 1, 256, 128))
        conv = self.convolutional(conv, (3, 3, 128, 256))
        conv = self.convolutional(conv, (1, 1, 256, 128))

        route_1 = conv
        conv = self.convolutional(conv, (3, 3, 128, 256))
        conv_sbbox = self.convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        conv = self.convolutional(route_1, (3, 3, 128, 256), downsample=True)
        conv = tf.concat([conv, route_2], axis=-1)

        conv = self.convolutional(conv, (1, 1, 512, 256))
        conv = self.convolutional(conv, (3, 3, 256, 512))
        conv = self.convolutional(conv, (1, 1, 512, 256))
        conv = self.convolutional(conv, (3, 3, 256, 512))
        conv = self.convolutional(conv, (1, 1, 512, 256))

        route_2 = conv
        conv = self.convolutional(conv, (3, 3, 256, 512))
        conv_mbbox = self.convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        conv = self.convolutional(route_2, (3, 3, 256, 512), downsample=True)
        conv = tf.concat([conv, route], axis=-1)

        conv = self.convolutional(conv, (1, 1, 1024, 512))
        conv = self.convolutional(conv, (3, 3, 512, 1024))
        conv = self.convolutional(conv, (1, 1, 1024, 512))
        conv = self.convolutional(conv, (3, 3, 512, 1024))
        conv = self.convolutional(conv, (1, 1, 1024, 512))

        conv = self.convolutional(conv, (3, 3, 512, 1024))
        conv_lbbox = self.convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        return [conv_sbbox, conv_mbbox, conv_lbbox]

    def call(self, input_, training=True, **kwargs):
        return self.yolo(input_)

    def get_config(self):
        config = {
            'num_classes': self.num_classes,
            'version': self.version,
            'use_weights': self.use_weights,
            'save_weights': self.save_weights
        }
        base_config = super(PretrainedYOLO, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        return [(None, 52, 52, 3 * (5 + self.num_classes)),
                (None, 26, 26, 3 * (5 + self.num_classes)),
                (None, 13, 13, 3 * (5 + self.num_classes))]


class PretrainedModel(Layer):

    def __init__(self, model_path: str = "", load_weights: bool = True, froze_model: bool = False, **kwargs):
        super(PretrainedModel, self).__init__(**kwargs)
        self.model_path = model_path
        self.load_weights = load_weights
        self.froze_model = froze_model
        self.model = None
        self.input_shapes = []
        self.output_shapes = []
        if self.model_path:
            self.system_path = os.path.join(self.model_path, "model")
            self.custom_obj_json = "trained_model_custom_obj_json.trm"
            self.model_json = "trained_model_json.trm"
            self.model_weights = "trained_model_weights"
            with os.scandir(self.system_path) as files:
                for f in files:
                    if 'generator_json' in f.name:
                        self.model_json = "generator_json.trm"
                    if 'generator_weights' in f.name:
                        self.model_weights = "generator_weights.trm"

            self.file_path_model_json = os.path.join(self.system_path, self.model_json)
            self.file_path_custom_obj_json = os.path.join(self.system_path, self.custom_obj_json)
            self.file_path_model_weights = os.path.join(self.system_path, self.model_weights)

            self.load()
            for inp in self.model.inputs:
                self.input_shapes.append(tuple(inp.shape))
            for outp in self.model.outputs:
                self.output_shapes.append(tuple(outp.shape))
            # logger.debug(f"self.input_shapes - {self.input_shapes}")
            # logger.debug(f"self.model.outputs - {self.output_shapes}")
            if self.load_weights:
                self.load_model_weights()
            if self.froze_model:
                for layer in self.model.layers:
                    layer.trainable = False
            else:
                for layer in self.model.layers:
                    layer.trainable = True

    def __get_json_data(self):
        with open(self.file_path_model_json) as json_file:
            data = json.load(json_file)

        with open(self.file_path_custom_obj_json) as json_file:
            custom_dict = json.load(json_file)

        return data, custom_dict

    @staticmethod
    def __set_custom_objects(custom_dict):
        custom_object = {}
        for k, v in custom_dict.items():
            try:
                custom_object[k] = getattr(importlib.import_module(f".{v}", package="terra_ai.custom_objects"), k)
            except:
                continue
        return custom_object

    def load(self) -> None:
        model_data, custom_dict = self.__get_json_data()
        custom_object = self.__set_custom_objects(custom_dict)
        self.model = tf.keras.models.model_from_json(model_data, custom_objects=custom_object)

    def load_model_weights(self):
        self.model.load_weights(self.file_path_model_weights)

    def call(self, input_, training=True, **kwargs):
        if self.model_path:
            return self.model(input_)
        else:
            return input_

    def get_config(self):
        config = {
            'model_path': self.model_path,
            'load_weights': self.load_weights,
            'froze_model': self.froze_model,
        }
        base_config = super(PretrainedModel, self).get_config()
        return dict(tuple(base_config.items()) + tuple(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        if self.model_path:
            return self.output_shapes
        else:
            return input_shape


if __name__ == "__main__":
    pass
