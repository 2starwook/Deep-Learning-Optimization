"""
Test optimizers using MNIST dataset from keras
"""

import time

import tensorflow as tf
from tensorflow import Module

from src.utils import load_np, cross_entropy_loss, accuracy, run_optimization
from src.model import ConvNet

TRAINING_STEPS = 200
BATCH_SIZE = 128
X_TRAIN_PATH = 'data/test/x_train.npy'
Y_TRAIN_PATH = 'data/test/y_train.npy'
X_TEST_PATH = 'data/test/x_test.npy'
Y_TEST_PATH = 'data/test/y_test.npy'


def run_test(optimizer: Module):
    x_train = load_np(X_TRAIN_PATH)
    y_train = load_np(Y_TRAIN_PATH)
    x_test = load_np(X_TEST_PATH)
    y_test = load_np(Y_TEST_PATH)

    # Use tf.data API to shuffle and batch data.
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.repeat().shuffle(5000).batch(BATCH_SIZE).prefetch(1)


    # Build neural network model.
    conv_net = ConvNet()

    # Run training for the given number of steps.
    times = list()
    loss_data = list()
    acc_data = list()
    acc_test_data = list()
    for step, (batch_x, batch_y) in enumerate(train_data.take(TRAINING_STEPS), 1):
        start = time.time()
        run_optimization(batch_x, batch_y, conv_net, optimizer)
        end = time.time()
        times.append(end-start)
        
        pred = conv_net(batch_x)
        test_pred = conv_net(x_test)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        acc_test = accuracy(test_pred, y_test)

        loss_data.append(loss)
        acc_data.append(acc)
        acc_test_data.append(acc_test)

    return times, loss_data, acc_data, acc_test_data