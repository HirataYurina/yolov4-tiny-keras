# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:yolo4_tiny.py
# software: PyCharm

import keras
import keras.layers as layers
import keras.backend as K
import tensorflow as tf


class Mish(layers.Layer):
    """Mish activation"""

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))


# this GroupFeature can not work because it cause some bugs:
# 'number of input channels does not match corresponding dimension of filter, 32!=64'
# class GroupFeature(layers.Layer):
#
#     def __init__(self, num_splits, **kwargs):
#         super(GroupFeature, self).__init__(**kwargs)
#         self.num_splits = num_splits
#
#     def call(self, inputs, **kwargs):
#         return tf.split(inputs, num_or_size_splits=self.num_splits, axis=-1)[0]


def conv_bn_activation(inputs,
                       filters,
                       filter_size,
                       downsample=False,
                       activation='leaky'):
    """yolo4-tiny is using leaky activation in source code"""

    assert activation in ['mish', 'leaky'], 'activation must be leaky or mish'

    if downsample:
        inputs = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(inputs)
        padding = 'valid'
        strides = 2
    else:
        padding = 'same'
        strides = 1

    x = layers.Conv2D(filters=filters,
                      kernel_size=filter_size,
                      strides=strides,
                      padding=padding,
                      use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(0.0005),
                      kernel_initializer=keras.initializers.normal(stddev=0.01),
                      bias_initializer=keras.initializers.constant(0.0))(inputs)
    x = layers.BatchNormalization()(x)
    if activation == 'mish':
        x = Mish()(x)
    else:
        x = layers.LeakyReLU(alpha=0.1)(x)

    return x


def csp_darknet_tiny(inputs):
    x = conv_bn_activation(inputs, 32, 3, downsample=True)
    x = conv_bn_activation(x, 64, 3, downsample=True)
    x = conv_bn_activation(x, 64, 3)

    route = x

    # TODO: the index is 1 in source code so i change 0 to 1
    x_group = layers.Lambda(lambda y: tf.split(y, num_or_size_splits=2, axis=-1)[1])(x)
    x_group = conv_bn_activation(x_group, 32, 3)
    route1 = x_group
    x_group = conv_bn_activation(x_group, 32, 3)
    x_group = layers.Concatenate()([x_group, route1])
    x_group = conv_bn_activation(x_group, 64, 1)
    x_group = layers.Concatenate()([route, x_group])
    x_group = layers.MaxPool2D(pool_size=2, padding='same')(x_group)

    x = conv_bn_activation(x_group, 128, 3)
    route = x

    # TODO: the index is 1 in source code so i change 0 to 1
    x_group = layers.Lambda(lambda y: tf.split(y, num_or_size_splits=2, axis=-1)[1])(x)
    x_group = conv_bn_activation(x_group, 64, 3)
    route1 = x_group
    x_group = conv_bn_activation(x_group, 64, 3)
    x_group = layers.Concatenate()([x_group, route1])
    x_group = conv_bn_activation(x_group, 128, 1)
    x_group = layers.Concatenate()([route, x_group])
    x_group = layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x_group)

    x = conv_bn_activation(x_group, 256, 3)
    route = x

    # TODO: the index is 1 in source code so i change 0 to 1
    x_group = layers.Lambda(lambda y: tf.split(y, num_or_size_splits=2, axis=-1)[1])(x)
    x_group = conv_bn_activation(x_group, 128, 3)
    route1 = x_group
    x_group = conv_bn_activation(x_group, 128, 3)
    x_group = layers.Concatenate()([x_group, route1])
    x_group = conv_bn_activation(x_group, 256, 1)

    C4 = x_group

    x_group = layers.Concatenate()([route, x_group])
    x_group = layers.MaxPool2D(pool_size=2, strides=2)(x_group)

    C5 = conv_bn_activation(x_group, 512, 3)

    return C4, C5


def yolo4_tiny(inputs, num_anchors, num_classes):

    C4, C5 = csp_darknet_tiny(inputs)

    x = conv_bn_activation(C5, 256, 1)
    output_C5 = conv_bn_activation(x, 512, 3)
    output_C5 = layers.Conv2D(num_anchors * (num_classes + 5),
                              1,
                              kernel_regularizer=keras.regularizers.l2(5e-4),
                              kernel_initializer=keras.initializers.normal(stddev=0.01),
                              bias_initializer=keras.initializers.constant(0.0))(output_C5)

    x_upsample = conv_bn_activation(x, 128, 1)
    x_upsample = layers.UpSampling2D()(x_upsample)
    x_concat = layers.Concatenate()([x_upsample, C4])

    output_C4 = conv_bn_activation(x_concat, 256, 3)
    output_C4 = layers.Conv2D(num_anchors * (5 + num_classes),
                              1,
                              kernel_regularizer=keras.regularizers.l2(5e-4),
                              kernel_initializer=keras.initializers.normal(stddev=0.01),
                              bias_initializer=keras.initializers.constant(0.0))(output_C4)

    model = keras.Model(inputs, [output_C4, output_C5])

    return model


if __name__ == '__main__':

    inputs = keras.Input(shape=(416, 416, 3))

    model = yolo4_tiny(inputs, num_anchors=3, num_classes=80)

    model.load_weights('../logs/yolo4_tiny_weight.h5')

    print(len(model.layers))

    # model.summary()

    # backbone = keras.Model(inputs, [output_C4, output_C5])

    # conv_names = []
    # bn_names = []
    #
    # for layer in backbone.layers:
    #     name = layer.name
    #     if name.startswith('conv2d'):
    #         conv_names.append(name)
    #     elif name.startswith('batch_normalization'):
    #         bn_names.append(name)
    #
    # print(conv_names)
    # print(len(conv_names))
