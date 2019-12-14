from __future__ import absolute_import, division, print_function, unicode_literals
from utils import *
import tensorflow as tf
from models import resnet_cifar_model

class Benchmark(object):
    def __init__(self, epochs, steps_per_epoch, model='resnet56'):
        self.model = None
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1,momentum=0.9, nesterov=True)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        self.callbacks = [tf.keras.callbacks.LearningRateScheduler(self.decay)]

    def create_model(self, model_name):
        if model_name == 'resnet56':
            self.model = resnet_cifar_model.resnet56(classes=10)
        else:
            err_msg = "Model name \"{}\" cannot be found!".format(model_name)
            print_msg(err_msg, 'err')
            sys.exit('Error!')
        return self.model

    def decay(self, epoch):
      if epoch < 3:
        return 1e-3
      elif epoch >= 3 and epoch < 7:
        return 1e-4
      else:
        return 1e-5

    def train_step(self, image, label):
        with tf.GradientTape() as tape:
            predictions = self.model(image, training=True)
            loss = self.loss(label, predictions)
            loss += sum(self.model.losses)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(label, predictions)

    def test_step(inputs):
        images, labels = inputs

        predictions = self.model(images, training=False)
        t_loss = self.loss(labels, predictions)

        test_loss.update_state(t_loss)
        test_accuracy.update_state(labels, predictions)

    def run(self, train_dataset, test_dataset, train_mode):
        if train_mode == 'loop':
            return self.loop_train(train_dataset, test_dataset)
        elif train_mode == 'fit':
            return self.fit_train(train_dataset.repeat(), test_dataset.repeat())

    def loop_train(self, train_dataset, test_dataset):
        header_str = ('Step\tImg/sec\ttotal_loss\taccuracy')
        print_msg(header_str, 'info')
        
        for epoch in range(self.epochs):
            for image, label in train_dataset:
                self.train_step(image, label)

                template = ('Epoch: {}, Train Loss: {}, Train Accuracy: {}, '
                              'Test Loss: {}, Test Accuracy: {}')

                print_msg(template.format(epoch, self.train_loss.result(),
                                      self.train_accuracy.result(),
                                      self.test_loss.result(),
                                      self.test_accuracy.result()), 'info', True)
            print_msg('')

    def compile_model(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])

    def fit_train(self, train_dataset, test_dataset):
        history = self.model.fit(train_dataset, epochs=self.epochs, steps_per_epoch=self.steps_per_epoch, callbacks=self.callbacks)
        return (history.history['loss'][-1],
                history.history['accuracy'][-1])

