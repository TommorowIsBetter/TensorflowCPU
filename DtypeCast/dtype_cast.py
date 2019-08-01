#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:Wang Yan
@ide:PyCharm
@time:2019/8/1 20:47
"""
import tensorflow as tf
a = tf.Variable([1, 0, 0, 1, 1])
b = tf.cast(a, dtype=tf.bool)
c = tf.Variable([1.22])
d = tf.cast(c, dtype=tf.int32)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(b))
print(sess.run(d))
