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

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 使用卷积网络
x_image = tf.reshape(x, [-1, 28, 28, 1])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME")


w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print("test accuracy %g" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob : 1.0}).eval())
