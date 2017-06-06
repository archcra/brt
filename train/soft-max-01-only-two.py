import tensorflow as tf
import loadTrainData

x_data = loadTrainData.getImagesData()
x_data = x_data[:2]


print(len(x_data), x_data[0][0], x_data[0][1], x_data[0][70*74-1], x_data[1][0], x_data[1][1], x_data[1][70*74-1])

x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], 
                                                        [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[1, 0], 
            [0, 1]]
            
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]


pixels = 4 # 70*74    
nb_classes = 3

        
X = tf.placeholder(tf.float32, [None, pixels]) # 图是70*74的
Y = tf.placeholder(tf.float32, [None, nb_classes])   # 目前共有7个图


W = tf.Variable(tf.random_normal([pixels, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# y_ = tf.matmul(x_data[0], W) + b
# print (y_)
# exit()

# Cross entropy cost/loss
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.maximum(hypothesis, 1e-13), axis=1))
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.clip_by_value(hypothesis, 1e-15, 1.0), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch graph
with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())

   for step in range(2001):
       sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
       if step % 200 == 0:
           print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
           
