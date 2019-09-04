#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Wang Yan
@ide: PyCharm
@Time : 2019/9/4 16:15
"""
import tensorflow as tf
tf.enable_eager_execution()
x = tf.random_uniform([3, 3])
print("Is there a GPU available: "),
print(tf.test.is_gpu_available())
print("Is the Tensor on GPU #0: "),
print(x.device.endswith('GPU:0'))
