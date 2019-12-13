from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets

class Dataset(object):
    def __init__(self, batch_size=64, height=32, width=32, channels=3, dataset='cifar10'):
        self.dataset = dataset
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.train_dataset = None
        self.test_dataset = None
        self.buffer_size = None
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.channels = channels

    def normalize(self, x, y):
        x = tf.image.per_image_standardization(x)
        return x, y

    def augmentation(self, x, y):
        x = tf.image.resize_with_crop_or_pad(x, HEIGHT + 8, WIDTH + 8)
        x = tf.image.random_crop(x, [self.height, self.width, self.channels])
        x = tf.image.random_flip_left_right(x)
        return x, y

    def create_dataset(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = datasets.cifar10.load_data()
        self.buffer_size = len(self.train_images)
        self.train_images = tf.cast(self.train_images, tf.float32)

        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.train_images, self.train_labels))
        self.test_dataset = tf.data.Dataset.from_tensor_slices((self.test_images, self.test_labels))


        # Preprocess and Shuffle the training data
        self.train_dataset = self.train_dataset.map(self.normalize).cache().shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=True)#.repeat()
        self.test_dataset = self.test_dataset.map(self.normalize).batch(self.batch_size, drop_remainder=True)#.repeat()

        return self.train_dataset, self.test_dataset

    def get_buffer_size(self):
        return int(self.buffer_size)

