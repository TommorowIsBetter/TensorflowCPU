#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Wang Yan
@ide: PyCharm
@Time : 2019/9/4 15:13
"""
from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.enable_eager_execution()
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([[1, 2, 3], [2, 3, 6]], 0))
print(tf.encode_base64("hello world"))
x = tf.matmul([[1]], [[2], [3]])
print(x.shape)
print(x.dtype)
