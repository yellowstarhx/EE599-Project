import os
import numpy as np
import tensorflow as tf
import csv
from tensorflow_vgg import vgg16
from sklearn.preprocessing import LabelBinarizer
from tensorflow_vgg import utils
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.misc import imread, imresize

data_dir = './pokemon_original_photos/'
contents = os.listdir(data_dir) 
classes = [each for each in contents if os.path.isdir(data_dir + each)] 
print(classes)

batch_size = 10
codes_list = []
labels = []
batch = []
codes = None
with tf.Session() as sess:
    
    vgg = vgg16.Vgg16()
    input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
    with tf.name_scope('content_vgg'):
        vgg.build(input_)
    for each in classes:
        print("Starting {} images".format(each))
        class_path = data_dir + each
        files = os.listdir(class_path)
        
        for ii, file in enumerate(files, 1):
            img = utils.load_image(os.path.join(class_path, file))
            batch.append(img.reshape((1, 224, 224, 3)))
            labels.append(each)
            
            if ii % batch_size == 0 or ii == len(files):
                images = np.concatenate(batch)  # [n.224,224,3]
                feed_dict = {input_: images}
                
                codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)  # codes_batch [n,4096]

                if codes is None:
                    codes = codes_batch
                else:
                    codes = np.concatenate((codes, codes_batch))

                batch = []
                print('{} images processed'.format(ii))

with open('codes', 'w') as f:
    codes.tofile(f)
with open('labels', 'w') as f:
    writer = csv.writer(f, delimiter='\n')
    writer.writerow(labels)
#############################prepare train set, validation set and test set#################################

lb = LabelBinarizer()
lb.fit(labels)
labels_vecs = lb.transform(labels)

# 8:1:1

ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)  
train_idx, val_idx = next(ss.split(codes, labels))

half_val_len = int(len(val_idx) / 2)
val_idx, test_idx = val_idx[:half_val_len], val_idx[half_val_len:]

train_x, train_y = codes[train_idx], labels_vecs[train_idx]
val_x, val_y = codes[val_idx], labels_vecs[val_idx]
test_x, test_y = codes[test_idx], labels_vecs[test_idx]

print('Train shapes(x,y):', train_x.shape, train_y.shape)
print("Validation shapes (x, y):", val_x.shape, val_y.shape)
print("Test shapes (x, y):", test_x.shape, test_y.shape)

###########################Train##########################################

inputs_ = tf.placeholder(tf.float32, shape=[None, codes.shape[1]])
print(input_.shape)
print(codes.shape[1])

labels_ = tf.placeholder(tf.int64, shape=[None, labels_vecs.shape[1]])
print(labels_.shape)
print(labels_vecs.shape[1])

fc = tf.contrib.layers.fully_connected(inputs_, 256)

logits = tf.contrib.layers.fully_connected(fc, labels_vecs.shape[1], activation_fn=None)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels_, logits=logits)

cost = tf.reduce_mean(cross_entropy)
# AdamOptimizer
optimizer = tf.train.AdamOptimizer().minimize(cost)

predicted = tf.nn.softmax(logits)

correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(labels_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def get_batches(x, y, n_batches=10):
    batch_size = len(x) // n_batches

    for ii in range(0, n_batches * batch_size, batch_size):
        
        if ii != (n_batches - 1) * batch_size:
            X, Y = x[ii: ii + batch_size], y[ii: ii + batch_size]
        else:
            X, Y = x[ii:], y[ii:]
        yield X, Y


epochs = 60
iteration = 0
saver=tf.train.Saver()
ckpt_dir = 'checkpoints/pokemon.ckpt'
# if not os.path.exists(ckpt_dir):
#     os.makedirs(ckpt_dir)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        # for x,y in get_batches(train_x,train_y):
        feed={inputs_:train_x,
              labels_:train_y}
        loss,_=sess.run([cost,optimizer],feed_dict=feed)
        print("Epoch: {}/{}".format(e + 1, epochs),
              "Iteration: {}".format(iteration),
              "Training loss: {:.5f}".format(loss))
        iteration += 1
        if iteration % 5 == 0:
            feed = {inputs_: val_x,
                    labels_: val_y}
            val_acc = sess.run(accuracy, feed_dict=feed)
            print("Epoch: {}/{}".format(e, epochs),
                  "Iteration: {}".format(iteration),
                  "Validation Acc: {:.4f}".format(val_acc))
        saver.save(sess, ckpt_dir)
#test
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

    feed = {inputs_: test_x,
            labels_: test_y}
    test_acc = sess.run(accuracy, feed_dict=feed)
    print("Test accuracy: {:.4f}".format(test_acc))
   #

# with tf.Session() as sess:
#     saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
#     imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
#     img1 = imread('./test_vgg/1/001.png', mode='RGB')
#     img1 = imresize(img1, (224, 224))

#     feed = {input_  : [img]}
#     prob = sess.run(predicted, feed_dict=feed)[0]
#     preds = (np.argsort(prob)[::-1])[0:5]
#     for p in preds:
#         print(classes[p], prob[p])
