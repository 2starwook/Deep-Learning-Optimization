import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras import Model
from tensorflow import Module


def smooth(data: list[float], weight: float = 0.9) -> list[float]:
    """
    data (List[float]): float data points
    weight (float, optional): range from 0 to 1 (default: 0.9)
    Ref. https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
    """
    last = data[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

def cross_entropy_loss(x, y):
    # Convert labels to int 64 for tf cross-entropy function.
    y = tf.cast(y, tf.int64)
    # Apply softmax to logits and compute cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    # Average loss across the batch.
    return tf.reduce_mean(loss)

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

def run_optimization(x, y, model: Model, optimizer: Module):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        # Forward pass.
        pred = model(x, is_training=True)
        # Compute loss.
        loss = cross_entropy_loss(pred, y)
        
    # Variables to update, i.e. trainable variables.
    trainable_variables = model.trainable_variables

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)
    
    # Update W and b following gradients.
    optimizer.apply_gradients(gradients, trainable_variables)


def save_np(file_name: str, object: np.ndarray):
    with open(file_name, 'wb') as fd:
        np.save(fd, object)

def load_np(file_name: str) -> np.ndarray:
    with open(file_name, 'rb') as fd:
        obj = np.load(fd)
    return obj

def store_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Convert to float32.
    x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
    # Normalize images value from [0, 255] to [0, 1].
    x_train, x_test = x_train / 255., x_test / 255.
    save_np('x_train.npy', x_train)
    save_np('y_train.npy', y_train)
    save_np('x_test.npy', x_test)
    save_np('y_test.npy', y_test)