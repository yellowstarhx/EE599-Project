import os
import numpy as np
import tensorflow as tf
import csv
from tensorflow_vgg import vgg16
from scipy.misc import imread, imresize
from tensorflow_vgg import utils

codes = None
data_dir = './pokemon_original_photos/'
contents = os.listdir(data_dir)
classes = [each for each in contents if os.path.isdir(data_dir + each)]
###########################train model##########################################

# input
inputs_ = tf.placeholder(tf.float32, shape=[None, 4096])
# labels
labels_ = tf.placeholder(tf.int64, shape=[None, 40])
# fully connected
fc = tf.contrib.layers.fully_connected(inputs_, 256)
# 5-dimensional 
logits = tf.contrib.layers.fully_connected(fc, 40, activation_fn=None)
# cross entropy
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels_, logits=logits)
# cost
cost = tf.reduce_mean(cross_entropy)
# AdamOptimizer
optimizer = tf.train.AdamOptimizer().minimize(cost)
# predicted class
predicted = tf.nn.softmax(logits)
# accuracy
correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(labels_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# save model
saver = tf.train.Saver()
with tf.Session() as sess:
    # build VGG 16
    vgg = vgg16.Vgg16()
    input_ = tf.placeholder(tf.float32, [1, 224, 224, 3])
    with tf.name_scope('content_vgg'):
        # load model
        vgg.build(input_)

        img = utils.load_image('./pokemon_original_photos/001/1.png')
        img1=img.reshape((1, 224, 224, 3))
        
        feed_dict = {input_: img1}
        
        codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)  # codes_batch [n,4096]
        
        codes = codes_batch
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
        feed = {inputs_: codes}
        prob = sess.run(predicted, feed_dict=feed)[0]
        preds = (np.argsort(prob)[::-1])[0:5]
        for p in preds:
            print(classes[p], prob[p])
