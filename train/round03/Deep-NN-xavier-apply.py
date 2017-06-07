#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import sys
import loadApplyData
import loadTrainData

pixels = 70*74    

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


W1 = tf.get_variable("W1", shape=[pixels, pixels],
     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([pixels]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1) # L1 就是原来的layer1


W2 = tf.get_variable("W2", shape=[pixels, 512],
                    initializer=tf.contrib.layers.xavier_initializer())
                    
b2 = tf.Variable(tf.random_normal([512]), name='bias2')
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)


W3 = tf.get_variable("W3", shape=[512, 512],
                    initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)


W4 = tf.get_variable("W4", shape=[512, 512],
                    initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)


W5 = tf.get_variable("W5", shape=[512, 7],
                    initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([7]))
hypothesis = tf.matmul(L4, W5) + b5


# Test model
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Test the model using test sets

imageName = 'data/Cr001.png'
testImage = loadApplyData.getImageData(imageName)
xData = [testImage]
testLabel = loadTrainData.getLabelData('c')
yData = [testLabel]
modelName = "saved/model-deep-nn-xavier.ckpt"

# Add ops to save and restore all the variables.
saver = tf.train.Saver()


# Launch graph
with tf.Session() as sess:
   # Restore variables
   saver.restore(sess, modelName)
   print("Accuracy: ", accuracy.eval(session=sess, 
        feed_dict={X: xData, Y: yData}))








