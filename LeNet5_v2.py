
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# load data
train_pokemon = pd.read_csv("/Users/yueyue/Desktop/usc/M2S2/EE599/project/new_train/train.csv")
test_pokemon = pd.read_csv("/Users/yueyue/Desktop/usc/M2S2/EE599/project/new_train/test.csv")


# In[3]:


#initialize size of training and testing data
num_train = 160
num_test = 40
num_output = 41


# In[4]:


np_train_pokemon = np.array(train_pokemon[1:num_train+1])
np_test_pokemon = np.array(test_pokemon[1:num_test+1])
test_image = np_test_pokemon[:,1:]
test_label = np_test_pokemon[:,0].astype('int64')
nb_classes = 41
test_label = np.eye(nb_classes)[test_label]


# In[5]:


sess = tf.InteractiveSession()

x = tf.placeholder('float', shape=[None, 28*28])
y_true = tf.placeholder('float', shape=[None, 41])
x_image = tf.reshape(x, [-1, 28, 28, 1])


def weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 1st layer: conv+relu+max_pool
w_conv1 = weights([5, 5, 1, 6])
b_conv1 = bias([6])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 2nd layer: conv+relu+max_pool
w_conv2 = weights([3, 3, 6, 16])
b_conv2 = bias([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])

# 3rd layer: 3*full connection
w_fc1 = weights([7*7*16, 120])
b_fc1 = bias([120])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1)+b_fc1)

w_fc2 = weights([120, 84])
b_fc2 = bias([84])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2)+b_fc2)

w_fc3 = weights([84, 41])
b_fc3 = bias([41])
h_fc3 = tf.nn.softmax(tf.matmul(h_fc2, w_fc3)+b_fc3)

cross_entropy = -tf.reduce_sum(y_true*tf.log(h_fc3))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(h_fc3, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

sess.run(tf.global_variables_initializer())

for i in range(60):
    X_batch = np_train_pokemon[:,1:]
    y_batch = np_train_pokemon[:,0].astype('int64')
    nb_classes = 41
    y_batch = np.eye(nb_classes)[y_batch]   
    if i%5 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={x: X_batch, y_true: y_batch})
        y_train_pred = sess.run(h_fc3, feed_dict={x: X_batch})
        print('step {}, training accuracy: {}'.format(i, train_accuracy))
    train_step.run(session=sess, feed_dict={x: X_batch, y_true: y_batch})

print('test accuracy: {}'.format(accuracy.eval(session=sess, feed_dict={x: test_image, y_true:
    test_label})))
y_test_pred = sess.run(h_fc3, feed_dict={x: test_image})


# In[6]:


np.savetxt("/Users/yueyue/Desktop/usc/M2S2/EE599/project/new_train/label_train.csv",np_train_pokemon[:,0], delimiter=',')
prediction_result_train=np.argmax(y_train_pred, axis=1)+1
np.savetxt("/Users/yueyue/Desktop/usc/M2S2/EE599/project/new_train/train_prediction.csv",prediction_result_train-1, delimiter=',')
np.savetxt("/Users/yueyue/Desktop/usc/M2S2/EE599/project/new_train/label_test.csv",np_test_pokemon[:,0], delimiter=',')
prediction_result_test=np.argmax(y_test_pred, axis=1)+1
np.savetxt("/Users/yueyue/Desktop/usc/M2S2/EE599/project/new_train/test_prediction.csv",prediction_result_test-1, delimiter=',')

