# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:mish.py
# software: PyCharm

import keras.backend as K
import keras


class Mish(keras.layers.Layer):
    """mish activation

    Mish: A Self Regularized Non-Monotonic Activation Function
    https://arxiv.org/abs/1908.08681?context=stat
    relu, leaky relu and prelu some disadvantages that they are to hard because they are  linear activation.
    But mish is soft that can bring more information into neural network.

    Mish(x) = x * tanh(log(1 + e ^ x))

    """
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        results = inputs * K.tanh(K.softplus(inputs))
        return results
