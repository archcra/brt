#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import sys
import loadApplyData

chessmenType = {
    'a': [0, 0, 1],
    'b': [0, 1, 0],
    'c': [0, 1, 1],
    'k': [1, 0, 0],
    'n': [1, 0, 1],
    'p': [1, 1, 0],
    'r': [1, 1, 1],
}
    
if len(sys.argv) != 4:
    print('Usage: python Deep-NN-04-apply.py saved/model-deep-nn-02.ckpt  data/Cr001.png c，第一个参数是存储的训练模型数据： model Name，第二个参数是待识别的棋子图片，第三个参数是图片对应的fen字母')
    exit(0)

imageName = sys.argv[2]
modelName = sys.argv[1]

x_data = [1]
x_data[0] = loadApplyData.getImageData(imageName)

pixels = 70*74    
            
y_data = np.array([chessmenType[sys.argv[3]]], dtype=np.float32)
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


W1 = tf.Variable(tf.random_normal([pixels, pixels]), name='weight1')
b1 = tf.Variable(tf.random_normal([pixels]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([pixels, 100]), name='weight2')
b2 = tf.Variable(tf.random_normal([100]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([100, 100]), name='weight3')
b3 = tf.Variable(tf.random_normal([100]), name='bias3')
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random_normal([100, 3]), name='weight4')
b4 = tf.Variable(tf.random_normal([3]), name='bias4')
hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Add ops to save and restore all the variables.
saver = tf.train.Saver()


# Launch graph
with tf.Session() as sess:
   # Restore variables
   saver.restore(sess, modelName)

   # Accuracy report
   h, c, a = sess.run([hypothesis, predicted, accuracy],
                      feed_dict={X: x_data, Y: y_data})
   print("\nHypothesis: ", h, "\nPredicted: ", c, "\nExpected: ", chessmenType[sys.argv[3]], "\nAccuracy: ", a)









