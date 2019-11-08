#
# Project 2, Part A (Object Recognition) 2
#

import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle
import multiprocessing as mp
import pandas as pd


NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 100
batch_size = 128


seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

def load_data(file):
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  #python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    
    labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
    labels_[np.arange(labels.shape[0]), labels-1] = 1

    return data, labels_

def cnn(images, filter1, filter2):

    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
    
    # Conv 1
    # Maps an RGB image of size 32x32 into feature maps of size 24x24 and pools into feature maps of size 12x12
    W1 = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS, filter1], stddev=1.0/np.sqrt(NUM_CHANNELS*9*9)), name='weights_1')
    b1 = tf.Variable(tf.zeros([filter1]), name='biases_1')

    conv_1 = tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + b1)
    pool_1 = tf.nn.max_pool(conv_1, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_1')

    # Conv 2
    # Maps feature maps of size 12x12 to feature maps of size 8x8 and pools into feature maps of size 4x4
    W2 = tf.Variable(tf.truncated_normal([5, 5, filter1, filter2], stddev=1.0/np.sqrt(filter1*5*5)), name='weights_2')
    b2 = tf.Variable(tf.zeros([filter2]), name='biases_2')

    conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, W2, [1, 1, 1, 1], padding='VALID') + b2)
    pool_2 = tf.nn.max_pool(conv_2, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_2')

    dim = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value 
    pool_2_flat = tf.reshape(pool_2, [-1, dim])

    # Fully connected layer
    # Maps feature maps of size 4x4 into 300 features
    W3 = tf.Variable(tf.truncated_normal([dim, 300], stddev=1.0/np.sqrt(dim)), name='weights_3')
    b3 = tf.Variable(tf.zeros([300]), name='biases_3')
    fc_1 = tf.nn.relu(tf.matmul(pool_2_flat, W3) + b3)
	
    # Softmax layer
    W4 = tf.Variable(tf.truncated_normal([300, NUM_CLASSES], stddev=1.0/np.sqrt(300)), name='weights_4')
    b4 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_4')
    logits = tf.matmul(fc_1, W4) + b4

    return conv_1, pool_1, conv_2, pool_2, logits

def train(filters):

    trainX, trainY = load_data('data_batch_1')
    print(trainX.shape, trainY.shape)
    
    testX, testY = load_data('test_batch_trim')
    print(testX.shape, testY.shape)

    trainX = (trainX - np.min(trainX, axis = 0))/np.max(trainX, axis = 0)
    testX = (testX - np.min(testX, axis = 0))/np.max(testX, axis = 0)

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE*NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    x_reshape = tf.reshape(x, (-1, NUM_CHANNELS, IMG_SIZE, IMG_SIZE))
    x_transpose = tf.transpose(x_reshape, (0, 2, 3, 1))
    conv_1, pool_1, conv_2, pool_2, logits = cnn(x_transpose, filters[0], filters[1])

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    N = len(trainX)
    idx = np.arange(N)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_cost = [] 
        test_acc = []
        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]
            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_step.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})
            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
            print('filter1', filters[0], 'filter2', filters[1], 'epoch', e+1, 'test accuracy', test_acc[e])

        return test_acc


def main():

    filters = [[10,10], [10,20], [10,30], [10,40], [10,50], [10,60], [10,70], [10,80], [10,90], [10,100], [20,10], [20,20], [20,30], [20,40], [20,50], [20,60], [20,70], [20,80], [20,90], [20,100], \
    [30,10], [30,20], [30,30], [30,40], [30,50], [30,60], [30,70], [30,80], [30,90], [30,100], [40,10], [40,20], [40,30], [40,40], [40,50], [40,60], [40,70], [40,80], [40,90], [40,100], \
    [50,10], [50,20], [50,30], [50,40], [50,50], [50,60], [50,70], [50,80], [50,90], [50,100], [60,10], [60,20], [60,30], [60,40], [60,50], [60,60], [60,70], [60,80], [60,90], [60,100], \
    [70,10], [70,20], [70,30], [70,40], [70,50], [70,60], [70,70], [70,80], [70,90], [70,100], [80,10], [80,20], [80,30], [80,40], [80,50], [80,60], [80,70], [80,80], [80,90], [80,100], \
    [90,10], [90,20], [90,30], [90,40], [90,50], [90,60], [90,70], [90,80], [90,90], [90,100], [100,10], [100,20], [100,30], [100,40], [100,50], [100,60], [100,70], [100,80], [100,90], [100,100]]
    test_acc = []
    for i in range(len(filters)):
        test_acc.append(train(filters[i]))

    # Save the test accuracies obtained in csv file
    df = pd.DataFrame(test_acc)
    df.to_csv("number_of_features.csv")

    plt.figure(1)
    for i in range(len(filters)):
        plt.plot(range(epochs), test_acc[i], label='filters = {}'.format(filters[i]))
    plt.xlabel('number of epochs')
    plt.ylabel('test accuracy')
    plt.legend()
    plt.title('Test Accuracy vs. Number of Epochs')
    plt.savefig('./figures_part_A/part2_test_accuracy.png')
    plt.show()

if __name__ == '__main__':
    main()
