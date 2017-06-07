#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 使用重复棋子的多个图片训练
import tensorflow as tf
import numpy as np

import loadTrainData

# parameters
learning_rate = 0.001
training_epochs = 300

x_data = loadTrainData.getImagesData()

pixels = 70*74    
imagesClass = 11
            
# y_data = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=np.float32)
y_data = np.array(loadTrainData
.getLabelsData())

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


W5 = tf.get_variable("W5", shape=[512, imagesClass],
                    initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([imagesClass]))
hypothesis = tf.matmul(L4, W5) + b5

# cost/loss function
# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Test model
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Add ops to save and restore all the variables.
saver = tf.train.Saver()


# Launch graph
with tf.Session() as sess:
   # Initialize TensorFlow variables
   sess.run(tf.global_variables_initializer())
   
   for epoch in range(training_epochs):
       c, _ = sess.run([cost, optimizer], feed_dict={X: x_data, Y: y_data})
       print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(c))

   #end for
   save_path = saver.save(sess, "saved/model-deep-nn-xavier.ckpt")

  







