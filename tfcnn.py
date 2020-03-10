# Convolutional Neural Network
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt
import tfgpu
import tfmnist


class Cnn(Model):
    """Class of the Convolutional Neural Network model
    """
    def __init__(self, filters=32, kernel_size=3, strides=2, units=(128, 10), name='CNN', **kwargs):
        """Initialize
        :param filters: number of output filters in the convolution
        :param kernel_size: length of the convolution window
        :param strides: stride length of the convolution
        :param units: dimensionality of the output space: 2d tuple
        :param name: class name
        :param kwargs:
        """
        super(Cnn, self).__init__(name=name, **kwargs)
        self.conv1 = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, activation='relu')
        self.flatten = Flatten()
        self.dens1 = Dense(units=units[0], activation='relu')
        self.dens2 = Dense(units=units[1], activation='softmax')

    def call(self, inputs, **kwargs):
        """Call model
        :param inputs: original space
        :param kwargs:
        :return: predicted space
        """
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.dens1(x)
        z = self.dens2(x)
        return z


@tf.function
def train_step(x, y, model, loss_fn, optimizer, loss_metrics, accu_metrics):
    """Trainiing step
    :param x: original data
    :param y: labels
    :param model:
    :param loss_fn:
    :param optimizer:
    :param loss_metrics: loss metrics
    :param accu_metrics: accuracy metrics
    """
    with tf.GradientTape() as tape:
        pred = model(x)
        loss_value = loss_fn(y, pred)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    loss_metrics(loss_value)
    accu_metrics(y, pred)


@tf.function
def test_step(x, y, model, loss_fn, loss_metrics, accu_metrics):
    """Test step
    :param x: original data
    :param y: labels
    :param model:
    :param loss_fn:
    :param loss_metrics: loss metrics
    :param accu_metrics: accuracy metrics
    :return: predicted space
    """
    z = model(x)
    loss_value = loss_fn(y, z)
    loss_metrics(loss_value)
    accu_metrics(y, z)
    return z


def perform_training(x_data, y_data, model, loss_fn, optimizer, batch_size=64, epochs=10):
    """Perform training
    :param x_data: original data
    :param y_data: labels
    :param model:
    :param loss_fn:
    :param optimizer:
    :param batch_size: batch size
    :param epochs: number of epochs
    """
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).shuffle(x_data.shape[0]).batch(batch_size)
    loss_metrics = tf.keras.metrics.Mean(name='train_loss')
    accu_metrics = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    template = 'Epoch %s: Loss = %s, Accuracy = %s'
    for epoch in range(epochs):
        for x, y in dataset:
            train_step(x, y, model, loss_fn, optimizer, loss_metrics, accu_metrics)
        print(template % (epoch + 1,
                          loss_metrics.result(),
                          accu_metrics.result() * 100.0))
        loss_metrics.reset_states()
        accu_metrics.reset_states()


def perform_testing(x_data, y_data, model, loss_fn, batch_size=64):
    """Perform testing
    :param x_data: original data
    :param y_data: labels
    :param model:
    :param loss_fn:
    :param batch_size: batch size
    """
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(batch_size)
    loss_metrics = tf.keras.metrics.Mean(name='test_loss')
    accu_metrics = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    template = 'Loss = %s, Accuracy = %s'
    for x, y in dataset:
        test_step(x, y, model, loss_fn, loss_metrics, accu_metrics)
    print(template % (loss_metrics.result(),
                      accu_metrics.result() * 100.0))


def perform_prediction(x_data, y_data, model, loss_fn, batch_size=1):
    """Perform prediction
    :param x_data: original data
    :param y_data: labels
    :param model:
    :param loss_fn:
    :param batch_size: batch size
    """
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(batch_size)
    loss_metrics = tf.keras.metrics.Mean(name='pred_loss')
    accu_metrics = tf.keras.metrics.SparseCategoricalAccuracy(name='pred_accuracy')
    shape = x_data.shape[1:3]
    labels = [str(i) for i in range(10)]
    image_title = 'Label = %d, Prod. = %d'
    graph_title = 'Loss = %.2f, Accuracy = %.2f'
    template = 'Number %s: Loss = %s, Accuracy = %s'
    for x, y in dataset:
        pred = test_step(x, y, model, loss_fn, loss_metrics, accu_metrics)
        prod_label = np.argmax(pred)
        if y != prod_label:
            loss = loss_metrics.result()
            accuracy = accu_metrics.result() * 100.0
            print(template % (y, loss, accuracy))
            plt.subplot(1, 2, 1)
            plt.axis('off')
            plt.title(image_title % (y, prod_label))
            plt.imshow(np.array(x).reshape(shape), cmap='gray')
            plt.subplot(1, 2, 2)
            plt.title(graph_title % (loss, accuracy))
            plt.bar(labels, pred[0], align='center')
            plt.show()
        loss_metrics.reset_states()
        accu_metrics.reset_states()


if __name__ == '__main__':
    # GPU activation or not.
    gpu_on = True
    # processing number
    #   0: training
    #   1: testing
    #   the others: prediction
    proc_num = 0

    tfgpu.initialize_gpu(gpu_on)    # Initialize GPU devices

    x_train, y_train, x_test, y_test = tfmnist.load_mnist()     # Load the MNIST dataset

    # Instantiate the model
    model = Cnn()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Perform training, testing or prediction
    if proc_num == 0:       # training
        perform_training(x_train, y_train, model, loss_fn, optimizer)
        model.save_weights('model', save_format='tf')
    elif proc_num == 1:     # testing
        model.load_weights('model')
        perform_testing(x_test, y_test, model, loss_fn)
    else:                   # prediction
        model.load_weights('model')
        perform_prediction(x_test, y_test, model, loss_fn)
