from __future__ import absolute_import, division, print_function, unicode_literals 
import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()

import os, json
import argparse

parser = argparse.ArgumentParser(description='Welcome')
parser.add_argument('-workers', type=str, default='dummy')
parser.add_argument('-type', type=str, default='dummy')
parser.add_argument('-index', type=str, default='dummy')

args = parser.parse_args()

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': args.workers.split(','),
    },
    'task': {'type': args.type, 'index': int(args.index)}
})


strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()



BUFFER_SIZE = 10000

# Scaling MNIST data from (0, 255] to (0., 1.]
def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255
  return image, label

datasets, info = tfds.load(name='mnist',
                           with_info=True,
                           as_supervised=True)

train_datasets_unbatched = datasets['train'].map(scale).shuffle(BUFFER_SIZE).repeat()
#train_datasets = train_datasets_unbatched.batch(BATCH_SIZE)


def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
  ])
  model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model



NUM_WORKERS = 2
BATCH_SIZE = 64
# Here the batch size scales up by number of workers since 
# `tf.data.Dataset.batch` expects the global batch size. Previously we used 64, 
# and now this becomes 128.
GLOBAL_BATCH_SIZE = 64 * NUM_WORKERS
train_datasets = train_datasets_unbatched.batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
with strategy.scope():
  multi_worker_model = build_and_compile_cnn_model()
multi_worker_model.fit(train_datasets, epochs=3, steps_per_epoch=60000//BATCH_SIZE)


