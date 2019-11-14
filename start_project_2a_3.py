#
# Project 2, Part A (Object Recognition) 3
#

import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle



NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 1000
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

def cnn(images, keep_prob):

    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
    
    # Conv 1
    W1 = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS, 90], stddev=1.0/np.sqrt(NUM_CHANNELS*9*9)), name='weights_1')
    b1 = tf.Variable(tf.zeros([90]), name='biases_1')

    conv_1 = tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + b1)
    pool_1 = tf.nn.max_pool(conv_1, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_1')

    pool1_drop = tf.nn.dropout(pool_1, keep_prob)

    # Conv 2
    W2 = tf.Variable(tf.truncated_normal([5, 5, 90, 100], stddev=1.0/np.sqrt(90*5*5)), name='weights_2')
    b2 = tf.Variable(tf.zeros([100]), name='biases_2')

    conv_2 = tf.nn.relu(tf.nn.conv2d(pool1_drop, W2, [1, 1, 1, 1], padding='VALID') + b2)
    pool_2 = tf.nn.max_pool(conv_2, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_2')

    pool2_drop = tf.nn.dropout(pool_2, keep_prob)

    dim = pool2_drop.get_shape()[1].value * pool2_drop.get_shape()[2].value * pool2_drop.get_shape()[3].value 
    pool_2_flat = tf.reshape(pool2_drop, [-1, dim])

    # Fully connected layer
    W3 = tf.Variable(tf.truncated_normal([dim, 300], stddev=1.0/np.sqrt(dim)), name='weights_3')
    b3 = tf.Variable(tf.zeros([300]), name='biases_3')
    fc_1 = tf.nn.relu(tf.matmul(pool_2_flat, W3) + b3)
    
    fc1_drop = tf.nn.dropout(fc_1, keep_prob)
	
    # Softmax layer
    W4 = tf.Variable(tf.truncated_normal([300, NUM_CLASSES], stddev=1.0/np.sqrt(300)), name='weights_4')
    b4 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_4')
    logits = tf.matmul(fc1_drop, W4) + b4

    return logits

def main():

    trainX, trainY = load_data('data_batch_1')
    print(trainX.shape, trainY.shape)
    
    testX, testY = load_data('test_batch_trim')
    print(testX.shape, testY.shape)

    trainX = (trainX - np.min(trainX, axis = 0))/np.max(trainX, axis = 0)
    testX = (testX - np.min(testX, axis = 0))/np.max(testX, axis = 0)

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE*NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])
    keep_prob = tf.placeholder(tf.float32)

    x_reshape = tf.reshape(x, (-1, NUM_CHANNELS, IMG_SIZE, IMG_SIZE))
    x_transpose = tf.transpose(x_reshape, (0, 2, 3, 1))

    logits = cnn(x_transpose, keep_prob)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    entropy_sum = tf.reduce_sum(cross_entropy)
    loss = tf.reduce_mean(cross_entropy)

    train_step1 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    train_step2 = tf.train.MomentumOptimizer(learning_rate, 0.1).minimize(loss)
    train_step3 = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    train_step4 = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    N = len(trainX)
    idx = np.arange(N)
    with tf.Session() as sess:

        print('GD')
        sess.run(tf.global_variables_initializer())
        train_cost_gd = [] 
        test_acc_gd = []
        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]
            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_step1.run(feed_dict={x: trainX[start:end], y_: trainY[start:end], keep_prob: 1.0})

            cost_sum = 0

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                cost_sum += entropy_sum.eval(feed_dict={x: trainX[start:end], y_: trainY[start:end], keep_prob: 1.0})

            train_cost = cost_sum/trainY.shape[0]
            train_cost_gd.append(train_cost)
            test_acc_gd.append(accuracy.eval(feed_dict={x: testX, y_: testY, keep_prob: 1.0}))
            print('epoch:', e+1, '| entropy:', train_cost_gd[e], '| test accuracy:', test_acc_gd[e])

        print('Adding Momentum Term')
        sess.run(tf.global_variables_initializer())
        train_cost_mom = [] 
        test_acc_mom = []
        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]
            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_step2.run(feed_dict={x: trainX[start:end], y_: trainY[start:end], keep_prob: 1.0})

            cost_sum = 0

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                cost_sum += entropy_sum.eval(feed_dict={x: trainX[start:end], y_: trainY[start:end], keep_prob: 1.0})

            train_cost = cost_sum/trainY.shape[0]
            train_cost_mom.append(train_cost)
            test_acc_mom.append(accuracy.eval(feed_dict={x: testX, y_: testY, keep_prob: 1.0}))
            print('epoch:', e+1, '| entropy:', train_cost_mom[e], '| test accuracy:', test_acc_mom[e])

        print('Using RMSProp Algorithm')
        sess.run(tf.global_variables_initializer())
        train_cost_rms = [] 
        test_acc_rms = []
        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]
            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_step3.run(feed_dict={x: trainX[start:end], y_: trainY[start:end], keep_prob: 1.0})

            cost_sum = 0

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                cost_sum += entropy_sum.eval(feed_dict={x: trainX[start:end], y_: trainY[start:end], keep_prob: 1.0})

            train_cost = cost_sum/trainY.shape[0]
            train_cost_rms.append(train_cost)
            test_acc_rms.append(accuracy.eval(feed_dict={x: testX, y_: testY, keep_prob: 1.0}))
            print('epoch:', e+1, '| entropy:', train_cost_rms[e], '| test accuracy:', test_acc_rms[e])

        print('Using Adam Optimizer')
        sess.run(tf.global_variables_initializer())
        train_cost_adam = [] 
        test_acc_adam = []
        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]
            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_step4.run(feed_dict={x: trainX[start:end], y_: trainY[start:end], keep_prob: 1.0})
           
            cost_sum = 0

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                cost_sum += entropy_sum.eval(feed_dict={x: trainX[start:end], y_: trainY[start:end], keep_prob: 1.0})
         
            train_cost = cost_sum/trainY.shape[0]
            train_cost_adam.append(train_cost)
            test_acc_adam.append(accuracy.eval(feed_dict={x: testX, y_: testY, keep_prob: 1.0}))
            print('epoch:', e+1, '| entropy:', train_cost_adam[e], '| test accuracy:', test_acc_adam[e])

        print('Adding Dropout')
        sess.run(tf.global_variables_initializer())
        train_cost_drop = [] 
        test_acc_drop = []
        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]
            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_step1.run(feed_dict={x: trainX[start:end], y_: trainY[start:end], keep_prob: 0.5})
           
            cost_sum = 0

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                cost_sum += entropy_sum.eval(feed_dict={x: trainX[start:end], y_: trainY[start:end], keep_prob: 1.0})
           
            train_cost = cost_sum/trainY.shape[0]
            train_cost_drop.append(train_cost)
            test_acc_drop.append(accuracy.eval(feed_dict={x: testX, y_: testY, keep_prob: 1.0}))
            print('epoch', e+1, 'entropy', train_cost_drop[e], 'test accuracy', test_acc_drop[e])

        plt.figure(1)
        plt.plot(range(epochs), train_cost_gd, label='GD')
        plt.plot(range(epochs), train_cost_mom, label='Momentum')
        plt.plot(range(epochs), train_cost_rms, label='RMS')
        plt.plot(range(epochs), train_cost_adam, label='Adam')
        plt.plot(range(epochs), train_cost_drop, label='Dropout')
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Entropy cost')
        plt.title('Entropy Cost on Training Data against Epochs')
        plt.legend()
        plt.savefig('./figures_part_A/part3_entropy_cost.png')

        plt.figure(2)
        plt.plot(range(epochs), test_acc_gd, label='GD')
        plt.plot(range(epochs), test_acc_mom, label='Momentum')
        plt.plot(range(epochs), test_acc_rms, label='RMS')
        plt.plot(range(epochs), test_acc_adam, label='Adam')
        plt.plot(range(epochs), test_acc_drop, label='Dropout')
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Test accuracy')
        plt.title('Test Acccuracy against Epochs')
        plt.legend()
        plt.savefig('./figures_part_A/part3_test_accuracy.png')

        plt.show()


if __name__ == '__main__':
  main()
