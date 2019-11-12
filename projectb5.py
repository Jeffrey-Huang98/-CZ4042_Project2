# -*- coding: utf-8 -*-
"""ProjectB5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Y5P6ZifJpRV5eM11N7hG_004DFiyl5vD
"""

import numpy as np
import pandas
import tensorflow as tf
import csv
import pylab as plt
import timeit

MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
FILTER_SHAPE11 = [20, 256]
FILTER_SHAPE12 = [20, 20]
FILTER_SHAPE2 = [20, 1]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15
EMBEDDING_SIZE = 20
HIDDEN_SIZE = 20

no_epochs = 100
lr = 0.01
batch_size = 128
keep_prob = 0.8

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def char_cnn_model(x, withDO):
  
  input_layer = tf.reshape(tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256, 1])

  with tf.variable_scope('CNN_0001'):
    conv1 = tf.layers.conv2d(
        input_layer,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE11,
        padding='VALID',
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')
    if withDO:
        pool1 = tf.nn.dropout(pool1, keep_prob)

  with tf.variable_scope('CNN_0002'):
    conv2 = tf.layers.conv2d(
        pool1,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE2,
        padding='VALID',
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
        conv2,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')
    if withDO:
        pool2 = tf.nn.dropout(pool2, keep_prob)
    pool2 = tf.squeeze(tf.reduce_max(pool2, 1), squeeze_dims=[1])

  logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)

  return input_layer, logits

def word_cnn_model(x, withDO):
  
  word_vectors = tf.contrib.layers.embed_sequence(x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

  word_list = tf.expand_dims(word_vectors, 3)

  with tf.variable_scope('CNN_1'):
    conv1 = tf.layers.conv2d(
        word_list,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE12,
        padding='VALID',
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')
    if withDO:
        pool1 = tf.nn.dropout(pool1, keep_prob)

  with tf.variable_scope('CNN_2'):
    conv2 = tf.layers.conv2d(
        pool1,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE2,
        padding='VALID',
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
        conv2,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')
    if withDO:
        pool2 = tf.nn.dropout(pool2, keep_prob)
    pool2 = tf.squeeze(tf.reduce_max(pool2, 1), squeeze_dims=[1])

  logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)

  return word_list, logits

def char_rnn_model(x, withDO):
  
  input_layer = tf.one_hot(x, 256)
  input_layer = tf.unstack(input_layer, axis=1)

  cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
  if withDO:
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
  _, encoding = tf.nn.static_rnn(cell, input_layer, dtype=tf.float32)

  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

  return input_layer, logits

def word_rnn_model(x, withDO):
  
  word_vectors = tf.contrib.layers.embed_sequence(x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
  word_list = tf.unstack(word_vectors, axis=1)

  cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
  if withDO:
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
  _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

  return word_list, logits

def read_data_chars():
  
  x_train, y_train, x_test, y_test = [], [], [], []

  with open('train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[1])
      y_train.append(int(row[0]))

  with open('test_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[1])
      y_test.append(int(row[0]))
  
  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(x_test)
  y_test = pandas.Series(y_test)
  
  
  char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
  x_train = np.array(list(char_processor.fit_transform(x_train)))
  x_test = np.array(list(char_processor.transform(x_test)))
  y_train = y_train.values
  y_test = y_test.values
  
  return x_train, y_train, x_test, y_test

def data_read_words():
  
  x_train, y_train, x_test, y_test = [], [], [], []
  
  with open('train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[2])
      y_train.append(int(row[0]))

  with open("test_medium.csv", encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[2])
      y_test.append(int(row[0]))
  
  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(x_test)
  y_test = pandas.Series(y_test)
  y_train = y_train.values
  y_test = y_test.values
  
  vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
      MAX_DOCUMENT_LENGTH)

  x_transform_train = vocab_processor.fit_transform(x_train)
  x_transform_test = vocab_processor.transform(x_test)

  x_train = np.array(list(x_transform_train))
  x_test = np.array(list(x_transform_test))

  no_words = len(vocab_processor.vocabulary_)
  print('Total words: %d' % no_words)

  return x_train, y_train, x_test, y_test, no_words

def main1(isCNN, withDO):
  tf.reset_default_graph()
  x_train, y_train, x_test, y_test = read_data_chars()

  print(len(x_train))
  print(len(x_test))

  # Create the model
  x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
  y_ = tf.placeholder(tf.int64)

  if(isCNN):
    inputs, logits = char_cnn_model(x, withDO)
  else:
    inputs, logits = char_rnn_model(x, withDO)

  # Optimizer
  predictions = tf.argmax(logits, axis=1)
  accuracy = tf.contrib.metrics.accuracy(predictions, y_)
  entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
  train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  start_time = timeit.default_timer()

  # training
  loss = []
  accuracies = []
  for e in range(no_epochs):
    cost = []
    num_minibatches = int(len(x_train) / batch_size)

    idx = np.arange(len(x_train))
    np.random.shuffle(idx)
    X_shuffled, Y_shuffled = x_train[idx], y_train[idx]

    for batch_offset in range(0, len(x_train), batch_size):
        batch_end = min(len(x_train), batch_offset + batch_size)
        minibatch_X, minibatch_Y = X_shuffled[batch_offset:batch_end], Y_shuffled[batch_offset:batch_end]
        _, loss_ = sess.run([train_op, entropy],feed_dict= {x:minibatch_X,y_:minibatch_Y})
        cost.append(loss_)
    loss_ = np.mean(cost)
    loss.append(loss_)

    # measure accuracy
    test_accuracy = sess.run(accuracy, feed_dict={x:x_test,y_:y_test})
    accuracies.append(test_accuracy)

    if e%1 == 0:
      print('iter: %d, entropy: %g, accuracy: %g'%(e, loss[e], accuracies[e]))
    
  end_time = timeit.default_timer()
  totaltime = end_time - start_time
  print(f"Finish training {no_epochs} epochs in {totaltime}s")
  
  sess.close()

  return accuracies, loss, totaltime

def main2(isCNN, withDO):
  tf.reset_default_graph()
  global n_words

  x_train, y_train, x_test, y_test, n_words = data_read_words()

  # Create the model
  x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
  y_ = tf.placeholder(tf.int64)

  if(isCNN):
    word_list, logits = word_cnn_model(x, withDO)
  else:
    word_list, logits = word_rnn_model(x, withDO)

  # Optimizer
  predictions = tf.argmax(logits, axis=1)
  accuracy = tf.contrib.metrics.accuracy(predictions, y_)
  entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
  train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  start_time = timeit.default_timer()

  # training
  loss = []
  accuracies = []
  for e in range(no_epochs):
    cost = []
    num_minibatches = int(len(x_train) / batch_size)

    idx = np.arange(len(x_train))
    np.random.shuffle(idx)
    X_shuffled, Y_shuffled = x_train[idx], y_train[idx]

    for batch_offset in range(0, len(x_train), batch_size):
        batch_end = min(len(x_train), batch_offset + batch_size)
        minibatch_X, minibatch_Y = X_shuffled[batch_offset:batch_end], Y_shuffled[batch_offset:batch_end]
        word_list_,_, loss_ = sess.run([word_list,train_op, entropy],feed_dict= {x:minibatch_X,y_:minibatch_Y})
        cost.append(loss_)
    loss_ = np.mean(cost)
    loss.append(loss_)
	
    # measure accuracy
    test_accuracy = sess.run(accuracy, feed_dict={x:x_test,y_:y_test})
    accuracies.append(test_accuracy)

    if e%1 == 0:
      print('iter: %d, entropy: %g, accuracy: %g'%(e, loss[e], accuracies[e]))
  
  end_time = timeit.default_timer()
  totaltime = end_time - start_time
  print(f"Finish training {no_epochs} epochs in {totaltime}s")
  
  sess.close()

  return accuracies, loss, totaltime

def main():
  accuracies1, loss1, totaltime1 = main1(True, False)
  accuracies2, loss2, totaltime2 = main2(True, False)
  accuracies3, loss3, totaltime3 = main1(False, False)
  accuracies4, loss4, totaltime4 = main2(False, False)

  accuraciesDO1, lossDO1, totaltimeDO1 = main1(True, True)
  accuraciesDO2, lossDO2, totaltimeDO2 = main2(True, True)
  accuraciesDO3, lossDO3, totaltimeDO3 = main1(False, True)
  accuraciesDO4, lossDO4, totaltimeDO4 = main2(False, True)

  totaltime = [totaltime1, totaltime2, totaltime3, totaltime4]
  totaltimeDO = [totaltimeDO1, totaltimeDO2, totaltimeDO3, totaltimeDO4]
  # plot learning curves
  plt.figure(1)
  plt.plot(range(no_epochs), accuracies1, label='CNN Char')
  plt.plot(range(no_epochs), accuracies2, label='CNN Words')
  plt.plot(range(no_epochs), accuracies3, label='RNN Char')
  plt.plot(range(no_epochs), accuracies4, label='RNN Words')
  plt.xlabel(str(no_epochs) + ' iterations')
  plt.ylabel('accuracy')
  plt.title('Accuracy against epochs')
  plt.legend()
  plt.savefig('./accuracy5.png')

  # plot learning curves
  plt.figure(2)
  plt.plot(range(no_epochs), loss1, label='CNN Char')
  plt.plot(range(no_epochs), loss2, label='CNN Words')
  plt.plot(range(no_epochs), loss3, label='RNN Char')
  plt.plot(range(no_epochs), loss4, label='RNN Words')
  plt.xlabel(str(no_epochs) + ' iterations')
  plt.ylabel('entropy')
  plt.title('Cost entropy against epochs')
  plt.legend()
  plt.savefig('./epochcost5.png')

  # plot learning curves
  plt.figure(3)
  plt.plot(totaltime)
  plt.xticks(range(4), ['CNN_Char', 'CNN_Words', 'RNN_Char', 'RNN_Words'])  
  plt.xlabel('model type')
  plt.ylabel('time')
  plt.title('Time Taken per Epochs for Different Neural Network Models')
  plt.savefig('./time5.png')

  # plot learning curves
  plt.figure(4)
  plt.plot(range(no_epochs), accuraciesDO1, label='CNN Char')
  plt.plot(range(no_epochs), accuraciesDO2, label='CNN Words')
  plt.plot(range(no_epochs), accuraciesDO3, label='RNN Char')
  plt.plot(range(no_epochs), accuraciesDO4, label='RNN Words')
  plt.xlabel(str(no_epochs) + ' iterations')
  plt.ylabel('accuracy')
  plt.title('Accuracy against epochs with Dropout')
  plt.legend()
  plt.savefig('./accuracyDO5.png')

  # plot learning curves
  plt.figure(5)
  plt.plot(range(no_epochs), lossDO1, label='CNN Char')
  plt.plot(range(no_epochs), lossDO2, label='CNN Words')
  plt.plot(range(no_epochs), lossDO3, label='RNN Char')
  plt.plot(range(no_epochs), lossDO4, label='RNN Words')
  plt.xlabel(str(no_epochs) + ' iterations')
  plt.ylabel('entropy')
  plt.title('Cost entropy against epochs with Dropout')
  plt.legend()
  plt.savefig('./epochcostDO5.png')

  # plot learning curves
  plt.figure(6)
  plt.plot(totaltime, label='without dropout')
  plt.plot(totaltimeDO, label='with dropout')
  plt.xticks(range(4), ['CNN_Char', 'CNN_Words', 'RNN_Char', 'RNN_Words'])  
  plt.xlabel('model type')
  plt.ylabel('time')
  plt.title('Time Taken per Epochs for Different Neural Network Models')
  plt.savefig('./timecomparison5.png')

  # plot learning curves
  plt.figure(7)
  plt.plot(range(no_epochs), accuracies1, label='CNN Char without dropout')
  plt.plot(range(no_epochs), accuracies2, label='CNN Words without dropout')
  plt.plot(range(no_epochs), accuracies3, label='RNN Char without dropout')
  plt.plot(range(no_epochs), accuracies4, label='RNN Words without dropout')
  plt.plot(range(no_epochs), accuraciesDO1, label='CNN Char with dropout')
  plt.plot(range(no_epochs), accuraciesDO2, label='CNN Words with dropout')
  plt.plot(range(no_epochs), accuraciesDO3, label='RNN Char with dropout')
  plt.plot(range(no_epochs), accuraciesDO4, label='RNN Words with dropout')
  plt.xlabel(str(no_epochs) + ' iterations')
  plt.ylabel('accuracy')
  plt.title('Accuracy against epochs')
  plt.legend()
  plt.savefig('./accuracycomparison5.png')

  # plot learning curves
  plt.figure(8)
  plt.plot(range(no_epochs), loss1, label='CNN Char without dropout')
  plt.plot(range(no_epochs), loss2, label='CNN Words without dropout')
  plt.plot(range(no_epochs), loss3, label='RNN Char without dropout')
  plt.plot(range(no_epochs), loss4, label='RNN Words without dropout')
  plt.plot(range(no_epochs), lossDO1, label='CNN Char with dropout')
  plt.plot(range(no_epochs), lossDO2, label='CNN Words with dropout')
  plt.plot(range(no_epochs), lossDO3, label='RNN Char with dropout')
  plt.plot(range(no_epochs), lossDO4, label='RNN Words with dropout')
  plt.xlabel(str(no_epochs) + ' iterations')
  plt.ylabel('entropy')
  plt.title('Cost entropy against epochs with Dropout')
  plt.legend()
  plt.savefig('./epochcostcomparison5.png')

if __name__ == '__main__':
  main()