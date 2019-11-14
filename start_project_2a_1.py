#
# Project 2, Part A (Object Recognition) 1
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

def cnn(images):

    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
    
    # Conv 1
    # Maps an RGB image of size 32x32 into 50 feature maps of size 24x24 and pools into 50 feature maps of size 12x12
    W1 = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS, 50], stddev=1.0/np.sqrt(NUM_CHANNELS*9*9)), name='weights_1')
    b1 = tf.Variable(tf.zeros([50]), name='biases_1')

    conv_1 = tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + b1)
    pool_1 = tf.nn.max_pool(conv_1, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_1')

    # Conv 2
    # Maps 50 feature maps of size 12x12 to 60 feature maps of size 8x8 and pools into 60 feature maps of size 4x4
    W2 = tf.Variable(tf.truncated_normal([5, 5, 50, 60], stddev=1.0/np.sqrt(50*5*5)), name='weights_2')
    b2 = tf.Variable(tf.zeros([60]), name='biases_2')

    conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, W2, [1, 1, 1, 1], padding='VALID') + b2)
    pool_2 = tf.nn.max_pool(conv_2, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_2')
    
    dim = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value 
    pool_2_flat = tf.reshape(pool_2, [-1, dim])
	
    # Fully connected layer
    # Maps 60 feature maps of size 4x4 into 300 features
    W3 = tf.Variable(tf.truncated_normal([dim, 300], stddev=1.0/np.sqrt(dim)), name='weights_3')
    b3 = tf.Variable(tf.zeros([300]), name='biases_3')
    fc_1 = tf.nn.relu(tf.matmul(pool_2_flat, W3) + b3)
	
    # Softmax layer
    W4 = tf.Variable(tf.truncated_normal([300, NUM_CLASSES], stddev=1.0/np.sqrt(300)), name='weights_4')
    b4 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_4')
    logits = tf.matmul(fc_1, W4) + b4

    return conv_1, pool_1, conv_2, pool_2, logits


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

    x_reshape = tf.reshape(x, (-1, NUM_CHANNELS, IMG_SIZE, IMG_SIZE))
    x_transpose = tf.transpose(x_reshape, (0, 2, 3, 1))
    conv_1, pool_1, conv_2, pool_2, logits = cnn(x_transpose)

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

            train_cost.append(loss.eval(feed_dict={x: trainX, y_: trainY}))
            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
            print('epoch', e+1, '| entropy:', train_cost[e], '| test accuracy:', test_acc[e])


        plt.figure()
        plt.plot(range(epochs), train_cost)
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Entropy cost')
        plt.title('Entropy Cost on Training Data against Epochs')
        plt.savefig('./figures_part_A/part1a_entropy_cost.png')

        plt.figure()
        plt.plot(range(epochs), test_acc)
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Test accuracy')
        plt.title('Test Acccuracy against Epochs')
        plt.savefig('./figures_part_A/part1a_test_accuracy.png')

        random_integers = np.random.randint(low=0, high=2000, size=2)
        random_integer_1 = random_integers[0]
        random_integer_2 = random_integers[1]

        X1 = testX[random_integer_1,:]
        X2 = testX[random_integer_2,:]

        #Test pattern 1
        plt.figure()
        plt.gray()
        X1_show = X1.reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE).transpose(1, 2, 0)
        plt.axis('off')
        plt.imshow(X1_show)
        plt.savefig('./figures_part_A/part1b_test1_inputimage.png')

        X1_conv1, X1_pool1, X1_conv2, X1_pool2 = sess.run([conv_1, pool_1, conv_2, pool_2], {x: X1.reshape(1,3072)})

        plt.figure()
        plt.gray()
        X1_conv1 = np.array(X1_conv1)
        for i in range(50):
        	plt.subplot(5, 10, i+1); plt.axis('off'); plt.imshow(X1_conv1[0,:,:,i])
        plt.savefig('./figures_part_A/part1b_test1_conv1.png')

        plt.figure()
        plt.gray()
        X1_pool1 = np.array(X1_pool1)
        for i in range(50):
        	plt.subplot(5, 10, i+1); plt.axis('off'); plt.imshow(X1_pool1[0,:,:,i])
        plt.savefig('./figures_part_A/part1b_test1_pool1.png')

        plt.figure()
        plt.gray()
        X1_conv2 = np.array(X1_conv2)
        for i in range(60):
        	plt.subplot(6, 10, i+1); plt.axis('off'); plt.imshow(X1_conv2[0,:,:,i])
        plt.savefig('./figures_part_A/part1b_test1_conv2.png')

        plt.figure()
        plt.gray()
        X1_pool2 = np.array(X1_pool2)
        for i in range(60):
        	plt.subplot(6, 10, i+1); plt.axis('off'); plt.imshow(X1_pool2[0,:,:,i])
        plt.savefig('./figures_part_A/part1b_test1_pool2.png')

        #Test pattern 2
        plt.figure()
        plt.gray()
        X2_show = X2.reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE).transpose(1, 2, 0)
        plt.axis('off')
        plt.imshow(X2_show)
        plt.savefig('./figures_part_A/part1b_test2_inputimage.png')

        X2_conv1, X2_pool1, X2_conv2, X2_pool2 = sess.run([conv_1, pool_1, conv_2, pool_2], {x: X2.reshape(1,3072)})

        plt.figure()
        plt.gray()
        X2_conv1 = np.array(X2_conv1)
        for i in range(50):
        	plt.subplot(5, 10, i+1); plt.axis('off'); plt.imshow(X2_conv1[0,:,:,i])
        plt.savefig('./figures_part_A/part1b_test2_conv1.png')

        plt.figure()
        plt.gray()
        X2_pool1 = np.array(X2_pool1)
        for i in range(50):
        	plt.subplot(5, 10, i+1); plt.axis('off'); plt.imshow(X2_pool1[0,:,:,i])
        plt.savefig('./figures_part_A/part1b_test2_pool1.png')

        plt.figure()
        plt.gray()
        X2_conv2 = np.array(X2_conv2)
        for i in range(60):
        	plt.subplot(6, 10, i+1); plt.axis('off'); plt.imshow(X2_conv2[0,:,:,i])
        plt.savefig('./figures_part_A/part1b_test2_conv2.png')

        plt.figure()
        plt.gray()
        X2_pool2 = np.array(X2_pool2)
        for i in range(60):
        	plt.subplot(6, 10, i+1); plt.axis('off'); plt.imshow(X2_pool2[0,:,:,i])
        plt.savefig('./figures_part_A/part1b_test2_pool2.png')

        # plt.show()



if __name__ == '__main__':
  main()
