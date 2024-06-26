import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout


class ConvNet(Model):
    # Set layers.
    def __init__(self):
        # MNIST dataset parameters.
        num_classes = 10 # total classes (0-9 digits).

        super(ConvNet, self).__init__()
        # Convolution Layer with 32 filters and a kernel size of 5.
        self.conv1 = Conv2D(32, kernel_size=5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool1 = MaxPool2D(2, strides=2)

        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv2 = Conv2D(64, kernel_size=3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool2 = MaxPool2D(2, strides=2)

        # Flatten the data to a 1-D vector for the fully connected layer.
        self.flatten = Flatten()

        # Fully connected layer.
        self.fc1 = Dense(1024)
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.dropout = Dropout(rate=0.5)

        # Output layer, class prediction.
        self.out = Dense(num_classes)

    # Set forward pass.
    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=is_training)
        x = self.out(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x
