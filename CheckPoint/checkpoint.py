#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:Wang Yan
@ide:PyCharm
@time:2019/7/30 10:54
"""
import tensorflow as tf
import numpy as np


def record_checkpoint():
    w = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name='weights')
    b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        save_path = saver.save(sess, "my_net/save_net.ckpt")
        print("Save to path:", save_path)


def restore_checkpoint():
    w = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
    b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "my_net/save_net.ckpt")
        print("weights:", sess.run(w))
        print("biases:", sess.run(b))


if __name__ == '__main__':
    restore_checkpoint()