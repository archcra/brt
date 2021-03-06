import tensorflow as tf
import numpy as np

import loadTrainData

x_data = loadTrainData.getImagesData()

pixels = 70*74    
            
y_data = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=np.float32)
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
   # Initialize TensorFlow variables
   sess.run(tf.global_variables_initializer())
   for step in range(101):
       sess.run(train, feed_dict={X: x_data, Y: y_data})
       if step % 100 == 0:
           print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run([W1, W2]))

   # Accuracy report
   h, c, a = sess.run([hypothesis, predicted, accuracy],
                      feed_dict={X: x_data, Y: y_data})
   print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)
   save_path = saver.save(sess, "saved/model-deep-nn.ckpt")









