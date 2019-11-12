# -*- coding: utf-8 -*-
"""ProjectB6A.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10iSq9SXs9jIlnUN44d48WQw0rbiqzxtA
"""

import numpy as np
import pandas
import tensorflow as tf
import csv
import pylab as plt

MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_LABEL = 15
EMBEDDING_SIZE = 20

no_epochs = 100
lr = 0.01
batch_size = 128

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def rnn_model_gru(x, isChar):
  if(isChar):
    input_layer = tf.one_hot(x, 256)
    input_layer = tf.unstack(input_layer, axis=1)
  else:
    word_vectors = tf.contrib.layers.embed_sequence(x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
    input_layer = tf.unstack(word_vectors, axis=1)

  cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
  _, encoding = tf.nn.static_rnn(cell, input_layer, dtype=tf.float32)

  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

  return input_layer, logits

def rnn_model(x, isChar):
  if(isChar):
    input_layer = tf.one_hot(x, 256)
    input_layer = tf.unstack(input_layer, axis=1)
  else:
    word_vectors = tf.contrib.layers.embed_sequence(x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
    input_layer = tf.unstack(word_vectors, axis=1)

  cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)
  output, encoding = tf.nn.static_rnn(cell, input_layer, dtype=tf.float32)
  
  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)
  
  return input_layer, logits

def rnn_model_lstm(x, isChar):
  if(isChar):
    input_layer = tf.one_hot(x, 256)
    input_layer = tf.unstack(input_layer, axis=1)
  else:
    word_vectors = tf.contrib.layers.embed_sequence(x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
    input_layer = tf.unstack(word_vectors, axis=1)

  cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)
  output, encoding = tf.nn.static_rnn(cell, input_layer, dtype=tf.float32)
 
  logits = tf.layers.dense(encoding[1], MAX_LABEL, activation=None)

  return input_layer, logits


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

def main1(isChar):
  tf.reset_default_graph()
  print("RNN GRU")
  if(isChar):
    x_train, y_train, x_test, y_test = read_data_chars()
  else:
    global n_words
    x_train, y_train, x_test, y_test, n_words = data_read_words()

  print(len(x_train))
  print(len(x_test))

  # Create the model
  x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
  y_ = tf.placeholder(tf.int64)

  inputs, logits = rnn_model_gru(x, isChar)

  # Optimizer
  predictions = tf.argmax(logits, axis=1)
  accuracy = tf.contrib.metrics.accuracy(predictions, y_)
  entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
  train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

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
  
  sess.close()

  return accuracies

def main2(isChar):
  tf.reset_default_graph()
  print("Basic RNN")
  if(isChar):
    x_train, y_train, x_test, y_test = read_data_chars()
  else:
    global n_words
    x_train, y_train, x_test, y_test, n_words = data_read_words()

  print(len(x_train))
  print(len(x_test))

  # Create the model
  x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
  y_ = tf.placeholder(tf.int64)

  inputs, logits = rnn_model(x, isChar)

  # Optimizer
  predictions = tf.argmax(logits, axis=1)
  accuracy = tf.contrib.metrics.accuracy(predictions, y_)
  entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
  train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

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
  
  sess.close()

  return accuracies

def main3(isChar):
  tf.reset_default_graph()
  print("RNN LSTM")
  if(isChar):
    x_train, y_train, x_test, y_test = read_data_chars()
  else:
    global n_words
    x_train, y_train, x_test, y_test, n_words = data_read_words()

  print(len(x_train))
  print(len(x_test))

  # Create the model
  x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
  y_ = tf.placeholder(tf.int64)

  inputs, logits = rnn_model_lstm(x, isChar)

  # Optimizer
  predictions = tf.argmax(logits, axis=1)
  accuracy = tf.contrib.metrics.accuracy(predictions, y_)
  entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
  train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

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
      print('iter: %d, entropy: %g, accuracy: %.10f'%(e, loss[e], accuracies[e]))
  
  sess.close()

  return accuracies

def main():
  accuraciesGRUChar = main1(True)
  accuraciesRNNChar = main2(True)
  accuraciesLSTMChar = main3(True)

  accuraciesGRUWord = main1(False)
  accuraciesRNNWord = main2(False)
  accuraciesLSTMWord = main3(False)

  # plot learning curves
  plt.figure(1)
  plt.plot(range(no_epochs), accuraciesGRUChar, label='accuracy GRU')
  plt.plot(range(no_epochs), accuraciesRNNChar, label='accuracy basic RNN')
  plt.plot(range(no_epochs), accuraciesLSTMChar, label='accuracy LSTM')
  plt.xlabel(str(no_epochs) + ' iterations')
  plt.ylabel('accuracy')
  plt.title('Accuracy against epochs for characters id')
  plt.legend()
  plt.savefig('./accuracy6aChar.png')

  # plot learning curves
  plt.figure(2)
  plt.plot(range(no_epochs), accuraciesGRUWord, label='accuracy GRU')
  plt.plot(range(no_epochs), accuraciesRNNWord, label='accuracy basic RNN')
  plt.plot(range(no_epochs), accuraciesLSTMWord, label='accuracy LSTM')
  plt.xlabel(str(no_epochs) + ' iterations')
  plt.ylabel('accuracy')
  plt.title('Accuracy against epochs for words id')
  plt.legend()
  plt.savefig('./accuracy6awords.png')

  # plot learning curves
  plt.figure(3)
  plt.plot(range(no_epochs), accuraciesGRUChar, label='accuracy GRU chars')
  plt.plot(range(no_epochs), accuraciesRNNChar, label='accuracy basic RNN chars')
  plt.plot(range(no_epochs), accuraciesLSTMChar, label='accuracy LSTM chars')
  plt.plot(range(no_epochs), accuraciesGRUWord, label='accuracy GRU words')
  plt.plot(range(no_epochs), accuraciesRNNWord, label='accuracy basic RNN words')
  plt.plot(range(no_epochs), accuraciesLSTMWord, label='accuracy LSTM words')
  plt.xlabel(str(no_epochs) + ' iterations')
  plt.ylabel('accuracy')
  plt.title('Accuracy against epochs')
  plt.legend()
  plt.savefig('./accuracy6a.png')

if __name__ == '__main__':
  main()