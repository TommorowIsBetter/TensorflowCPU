#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:Wang Yan
@ide:PyCharm
@time:2019/8/1 21:21
"""
import tensorflow as tf
prediction_value = tf.Variable([[1.2], [2], [3]], dtype=tf.float32)
real_value = tf.Variable([[1], [2], [3.3]], dtype=tf.float32)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction_value - real_value), reduction_indices=[1]))
init = tf.initialize_all_variables()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(loss))
