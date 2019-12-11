from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import json

from tensorflow import keras
import resnet_cifar_model
import time
import matplotlib.pyplot as plt

"""
Remember to set the TF_CONFIG envrionment variable.

For example:

export TF_CONFIG='{"cluster": {"worker": ["10.1.10.58:12345", "10.1.10.250:12345"]}, "task": {"index": 0, "type": "worker"}}'
"""

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        # use this value as reference to calculate cummulative time taken
        self.timetaken = time.clock()
    def on_epoch_end(self,epoch,logs = {}):
        speed_mean = epoch,time.clock() - self.timetaken
        self.times.append((epoch,time.clock() - self.timetaken))
        #log_str = 'images/sec: %.1f' % speed_mean
        print('################################################################')
        print(speed_mean)
    def on_train_end(self,logs = {}):
        plt.xlabel('Epoch')
        plt.ylabel('Total time taken until an epoch in seconds')
        plt.plot(*zip(*self.times))
        plt.show()



import datetime

class MyCustomCallback(tf.keras.callbacks.Callback):
  def __init__(self):
      self.times = []
      self.timetaken = time.clock()

  def on_train_batch_begin(self, batch, logs=None):
    print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_train_batch_end(self, batch, logs=None):
    print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))
    speed = time.clock() - self.timetaken
    print('Training: batch {} taks {}'.format(batch, speed))
  def on_test_batch_begin(self, batch, logs=None):
    print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_end(self, batch, logs=None):
    print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))


NUM_GPUS = 2
BS_PER_GPU = 128
NUM_EPOCHS = 1

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
NUM_TRAIN_SAMPLES = 50000

BASE_LEARNING_RATE = 0.1
LR_SCHEDULE = [(0.1, 30), (0.01, 45)]


def normalize(x, y):
  x = tf.image.per_image_standardization(x)
  return x, y


def augmentation(x, y):
    x = tf.image.resize_with_crop_or_pad(
        x, HEIGHT + 8, WIDTH + 8)
    x = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS])
    x = tf.image.random_flip_left_right(x)
    return x, y


def schedule(epoch):
  initial_learning_rate = BASE_LEARNING_RATE * BS_PER_GPU / 128
  learning_rate = initial_learning_rate
  for mult, start_epoch in LR_SCHEDULE:
    if epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
  return learning_rate


(x,y), (x_test, y_test) = keras.datasets.cifar10.load_data()

train_dataset = tf.data.Dataset.from_tensor_slices((x,y))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

tf.random.set_seed(22)
train_dataset = train_dataset.map(augmentation).map(normalize).shuffle(NUM_TRAIN_SAMPLES).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True).repeat()
test_dataset = test_dataset.map(normalize).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True).repeat()
input_shape = (HEIGHT, WIDTH, NUM_CHANNELS)
img_input = tf.keras.layers.Input(shape=input_shape)
#optimizer = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
optimizer = keras.optimizers.Adam()

@tf.function
def train(model, dataset, optimizer):
  for x, y in dataset:
    with tf.GradientTape() as tape:
      prediction = model(x)
      loss = loss_fn(prediction, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


timetaken = timecallback()
with strategy.scope():
  model = resnet_cifar_model.resnet56(classes=NUM_CLASSES)
  #model.compile(
  #          optimizer=optimizer,
  #          loss='sparse_categorical_crossentropy',
  #          metrics=['sparse_categorical_accuracy'])

  EPOCHS = 5
  for epoch in range(EPOCHS):
      for images, labels in train_dataset:
          train(model, optimizer, train_dataset)

  '''print('Running warm up')
  model.fit(train_dataset,epochs=1, steps_per_epoch=(BS_PER_GPU * NUM_GPUS), verbose=0)
  print('Done warm up')

  header_str = ('Step\tImg/sec\ttotal_loss\taccuracy')
  print(header_str)

  model.fit(train_dataset,
          epochs=NUM_EPOCHS, steps_per_epoch=50000//(BS_PER_GPU * NUM_GPUS), verbose=0, callbacks = [MyCustomCallback()])
  '''
