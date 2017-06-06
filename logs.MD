


-> ModuleNotFoundError: No module named 'PIL'

$ sudo -H pip3 install image



-> ModuleNotFoundError: No module named 'matplotlib'
$ sudo -H pip3 install matplotlib


-> Install TensorFlow on Mac
https://www.tensorflow.org/install/install_mac

TensorFlow with CPU support only. (没有 NVIDIA CUDA GPU，是Intel的显卡)

安装有四种方式：
virtualenv
"native" pip
Docker
installing from sources


使用推荐的virtualenv

Installing with virtualenv

zhoutianweis-MacBook-Pro:AI papa$ virtualenv --system-site-packages -p python3 $targetDirectory
Your PYTHONPATH points to a site-packages dir for Python 3.x but you are running Python 2.x!
     PYTHONPATH is currently: "/usr/local/lib/python3.6/site-packages"
     You should `unset PYTHONPATH` to fix this.
zhoutianweis-MacBook-Pro:AI papa$ export VIRTUALENV_PYTHON=/usr/local/lib/python3.6
zhoutianweis-MacBook-Pro:AI papa$ virtualenv --system-site-packages -p python3 $targetDirectory
Your PYTHONPATH points to a site-packages dir for Python 3.x but you are running Python 2.x!
     PYTHONPATH is currently: "/usr/local/lib/python3.6/site-packages"
     You should `unset PYTHONPATH` to fix this.
zhoutianweis-MacBook-Pro:AI papa$ unset PYTHONPATH
zhoutianweis-MacBook-Pro:AI papa$
zhoutianweis-MacBook-Pro:AI papa$
zhoutianweis-MacBook-Pro:AI papa$ virtualenv --system-site-packages -p python3 $targetDirectory
Running virtualenv with interpreter /usr/local/bin/python3
Using base prefix '/usr/local/Cellar/python3/3.6.1/Frameworks/Python.framework/Versions/3.6'
New python executable in /Users/papa/tensorflow/bin/python3.6
Also creating executable in /Users/papa/tensorflow/bin/python
Installing setuptools, pip, wheel...

Installing setuptools, pip, wheel...done.

$ source ~/tensorflow/bin/activate


(tensorflow) zhoutianweis-MacBook-Pro:AI papa$


开始安装：
pip3 install --upgrade tensorflow


在虚拟环境中：
python

>>> import tensorflow as tf
>>>


下次使用：
source ~/tensorflow/bin/activate



-> Op has type float32 that does not match type int32 of argument 'a'



当仅处理两个图时，误差为：

import tensorflow as tf
import loadTrainData

x_data = loadTrainData.getImagesData()
x_data = x_data[:2]
print(len(x_data), x_data[0][0], x_data[0][1], x_data[0][70*74-1], x_data[1][0], x_data[1][1], x_data[1][70*74-1])

y_data = [[1, 0], 
            [0, 1]]
pixels = 70*74            
X = tf.placeholder(tf.float32, [None, pixels]) # 图是70*74的
Y = tf.placeholder(tf.float32, [None, 2])   # 目前共有7个图
nb_classes = 2

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
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.maximum(hypothesis, 1e-5), axis=1))


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch graph
with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())

   for step in range(2001):
       sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
       if step % 200 == 0:
           print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
           



2000 -0.500005

需要注意，这里需要处理一下：
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.maximum(hypothesis, 1e-5), axis=1))

否则误差为Nan


修改为原数据后，将1e-5去掉，结果是：
200 0.56964
400 0.471067
600 0.39494
800 0.323574
1000 0.25343
1200 0.224954
1400 0.204857
1600 0.187923
1800 0.173469
2000 0.160997

否则是：
0 -0.406157
200 -0.604369
400 -0.638502
600 -0.659712
800 -0.674137
1000 -0.684739
1200 -0.692932
1400 -0.699462
1600 -0.704777
1800 -0.709174
2000 -0.712859

这个当从训练7个数据，减到2个时，好像有提高。虽然感觉是蒙事儿。因为那么多点，使用一层nn，应该根本找不出区别出来。



采用Deep NN for XOR的方法：


这个学习2个值，精度为1：

Hypothesis:  [[ 0.13637529]
 [ 0.84035379]]
Correct:  [[ 0.]
 [ 1.]]
Accuracy:  1.0


代码如下：
import tensorflow as tf
import numpy as np

import loadTrainData

x_data = loadTrainData.getImagesData()
x_data = x_data[:2]
            
pixels = 70*74    

            
y_data = np.array([[0], [1]], dtype=np.float32)
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


W1 = tf.Variable(tf.random_normal([pixels, pixels]), name='weight1')
b1 = tf.Variable(tf.random_normal([pixels]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([pixels, 10]), name='weight2')
b2 = tf.Variable(tf.random_normal([10]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([10, 10]), name='weight3')
b3 = tf.Variable(tf.random_normal([10]), name='bias3')
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random_normal([10, 1]), name='weight4')
b4 = tf.Variable(tf.random_normal([1]), name='bias4')
hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
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












学习7个时，精度为：Accuracy:  0.666667



代码如下：

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

W2 = tf.Variable(tf.random_normal([pixels, 10]), name='weight2')
b2 = tf.Variable(tf.random_normal([10]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([10, 10]), name='weight3')
b3 = tf.Variable(tf.random_normal([10]), name='bias3')
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random_normal([10, 3]), name='weight4')
b4 = tf.Variable(tf.random_normal([3]), name='bias4')
hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
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








第三个nn，将中间层节点都增加到100，精度为1：

Hypothesis:  [[ 0.02854439  0.03971684  0.95990711]
 [ 0.08871716  0.8145116   0.00455966]
 [ 0.06952807  0.91531903  0.94405496]
 [ 0.9238615   0.15650237  0.11028335]
 [ 0.91997683  0.13394679  0.97246373]
 [ 0.99110669  0.99235898  0.02895392]
 [ 0.97899717  0.9721781   0.97675109]]
Correct:  [[ 0.  0.  1.]
 [ 0.  1.  0.]
 [ 0.  1.  1.]
 [ 1.  0.  0.]
 [ 1.  0.  1.]
 [ 1.  1.  0.]
 [ 1.  1.  1.]]
Accuracy:  1.0


-> save  Variables & loading

https://www.tensorflow.org/programmers_guide/variables


这个训练不到1分钟，不过保存的东西好在啊，100多M:
(tensorflow) zhoutianweis-MacBook-Pro:train papa$ ls -al saved/
total 213912
drwxr-xr-x   6 papa  staff        204 Jun  6 20:01 .
drwxr-xr-x  17 papa  staff        578 Jun  6 19:59 ..
-rw-r--r--   1 papa  staff         93 Jun  6 20:01 checkpoint
-rw-r--r--   1 papa  staff  109464332 Jun  6 20:01 model-deep-nn.ckpt.data-00000-of-00001
-rw-r--r--   1 papa  staff        328 Jun  6 20:01 model-deep-nn.ckpt.index
-rw-r--r--   1 papa  staff      46001 Jun  6 20:01 model-deep-nn.ckpt.meta



使用loader后，不训练也能得到1:


Hypothesis:  [[ 0.09198416  0.04148368  0.94446903]
 [ 0.03038336  0.99210417  0.00901892]
 [ 0.0319663   0.92764282  0.97463214]
 [ 0.96464533  0.10807684  0.03376865]
 [ 0.97010291  0.03378425  0.99479717]
 [ 0.94865799  0.92305768  0.09510436]
 [ 0.97063553  0.96710837  0.96720761]]
Correct:  [[ 0.  0.  1.]
 [ 0.  1.  0.]
 [ 0.  1.  1.]
 [ 1.  0.  0.]
 [ 1.  0.  1.]
 [ 1.  1.  0.]
 [ 1.  1.  1.]]
Accuracy:  1.0


代码如下：


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
   # Restore variables
   saver.restore(sess, "saved/model-deep-nn.ckpt")

   # Accuracy report
   h, c, a = sess.run([hypothesis, predicted, accuracy],
                      feed_dict={X: x_data, Y: y_data})
   print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)
















