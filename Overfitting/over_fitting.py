#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:Wang Yan
@ide:PyCharm
@time:2019/7/30 20:07
"""
import tensorflow as tf


def over_fitting():
    fc = tf.Variable([1, 2, 3, 4], dtype=tf.float32, name='fc')
    h_fc1_drop = tf.nn.dropout(fc, 0.25)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(h_fc1_drop))


if __name__ == '__main__':
    over_fitting()
