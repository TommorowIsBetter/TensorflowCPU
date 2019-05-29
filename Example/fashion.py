#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:Wang Yan
@ide:PyCharm
@time:2019/5/29 11:17
"""
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# 以下内容显示训练集中有 60000 张图像，每张图像都表示为 28x28 像素
print(train_images.shape)
# 训练集中有6000个标签
print(len(train_labels))
# 每个标签都是一个介于 0 到 9 之间的整数：
print(train_labels)
# 测试集的数据格式
print(test_images.shape)
# 测试集有1000个标签
print(len(test_labels))
# 测试集数据标签
print(test_labels)

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# 我们将这些值缩小到0到1之间，然后将其馈送到神经网络模型
train_images = train_images / 255.0
test_images = test_images / 255.0
# 显示训练集中的前 25 张图像，并在每张图像下显示类别名称
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# 神经网络的层
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
# 编译模型
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型，使模型与数据拟合
model.fit(train_images, train_labels, epochs=10)
# 评估准确率
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
# 做出预测
predictions = model.predict(test_images)
# 查看第一个预测值
print(predictions[0])
# 预测结果是一个具有 10 个数字的数组。这些数字说明模型对于图像对应于 10 种不同服饰中每一个服饰的“置信度”
print(np.argmax(predictions[0]))
# 打印实际的值
print(test_labels[0])










