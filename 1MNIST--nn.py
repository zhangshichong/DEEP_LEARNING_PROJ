from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import scipy.misc
import os
import numpy as np
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

save_dir = "D:\codes\\tensorflow-projects\MNIST_data\\raw\\"
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)
for i in range(20):
    image_array = mnist.train.images[i,:]
    image_array = image_array.reshape(28, 28)
    file_name = save_dir+'mnist_train_%d.jpg'%i
    scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(file_name)
    one_hot_label = mnist.train.labels[i, :]
    label = np.argmax(one_hot_label)
    # print('mnist_train_%d.jpg label: %d' % (i, label))
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 模型输出
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 实际标签
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_: mnist.test.labels}))
