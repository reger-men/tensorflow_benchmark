from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras import datasets, layers, models
import resnet_cifar_model


NUM_TRAIN_SAMPLES = 50000
BS_PER_GPU = 128
NUM_GPUS = 2
HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10

def normalize(x, y):
    x = tf.cast(x, tf.float32)
    x = tf.image.per_image_standardization(x)
    return x, y

def augmentation(x, y):
    x = tf.cast(x, tf.float32)
    x = tf.image.resize_with_crop_or_pad(x, HEIGHT + 8, WIDTH + 8)
    x = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS])
    x = tf.image.random_flip_left_right(x)
    return x, y


class Benchmark(object):
    def __init__(self, epochs, model=None):
        self.model = model
        self.epochs = epochs
        self.global_batch_size = 128 #global_batch_size
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1,momentum=0.9, nesterov=True)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


    def compute_loss(self, labels, predictions):
        per_example_loss = self.loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.global_batch_size)

    def train_step(self, image, label):
        with tf.GradientTape() as tape:
            predictions = self.model(image, training=True)
            loss = self.loss_object(label, predictions)
            loss += sum(self.model.losses)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(label, predictions)

    def test_step(inputs):
        images, labels = inputs

        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss.update_state(t_loss)
        test_accuracy.update_state(labels, predictions)

    def run(self, train_dataset, test_dataset):
        for epoch in range(self.epochs):
            for image, label in train_dataset:
                self.train_step(image, label)

        template = ('Epoch: {}, Train Loss: {}, Train Accuracy: {}, '
                      'Test Loss: {}, Test Accuracy: {}')

        print(template.format(epoch, self.train_loss.result(),
                              self.train_accuracy.result(),
                              self.test_loss.result(),
                              self.test_accuracy.result()))


    
class Dataset(object):
    def __init__(self, dataset=None):
        self.dataset = dataset
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None

    def create_dataset(self): 
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = datasets.cifar10.load_data()
        self.train_images = tf.cast(self.train_images, tf.float32)


#BUFFER_SIZE = len(train_images)
BATCH_SIZE_PER_REPLICA = 128
NUM_GPUS = 2
BS_PER_GPU = 64 
#GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = 1

# Download and prepare the CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    
    
tf.random.set_seed(22)
train_dataset = train_dataset.map(augmentation).map(normalize).shuffle(NUM_TRAIN_SAMPLES).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)#.repeat()
test_dataset = test_dataset.map(normalize).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)#.repeat()

# Normalize pixel values to be between 0 and 1
#train_images, test_images = train_images / 255.0, test_images / 255.0

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

strategy = tf.distribute.MirroredStrategy()
print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

#train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
#test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

model = resnet_cifar_model.resnet56(classes=10)
optimizer = tf.keras.optimizers.Adam()


train_obj = Benchmark(1, model)
print('Training...')
train_obj.run(train_dataset, test_dataset)
print('Training Done.')

























def print_msg(typ, msg):
    """Print msg in specific format by it type"""

    TXTEND = '\033[0m'
    TXTBOLD = '\033[1m'
    if typ == 'info':
        msg = TXTBOLD + '\033[95m' + msg + TXTEND
    elif typ == 'succ':
        msg = TXTBOLD + '\033[92m' + msg + TXTEND
    elif typ == 'warn':
        msg = TXTBOLD + '\033[93m' + msg + TXTEND
    elif typ == 'err':
        msg = TXTBOLD + '\033[91m' + msg + TXTEND
    else:
        msg = msg

    print(msg)

