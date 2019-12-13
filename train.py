from __future__ import absolute_import, division, print_function, unicode_literals

from absl import app
from absl import flags
from utils import *

import tensorflow as tf
print_msg(tf.__version__, 'info')

# Filter out INFO & WARNING messages
tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(2)


from tensorflow.keras import datasets, layers, models
from benchmark import Benchmark
from data import Dataset


FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 1, 'Number of epochs')
flags.DEFINE_integer('buffer_size', 50000, 'Shuffle buffer size')
flags.DEFINE_integer('batch_size', 64, 'Batch Size')
flags.DEFINE_string('train_mode', 'loop', 'Use either keras fit or loop training')
flags.DEFINE_string('distribution_strategy', 'OneDevice', 'Can be: Mirrored, MultiWorker, OneDevice')
flags.DEFINE_integer('num_gpus', 1, 'Number of GPUs. 0 will run on CPU')

def run_main(argv):
    del argv
    kwargs = {
            'epochs': FLAGS.epochs,
            'buffer_size': FLAGS.buffer_size,
            'batch_size': FLAGS.batch_size,
            'train_mode': FLAGS.train_mode,
            'distribution_strategy': FLAGS.distribution_strategy,
            'num_gpus': FLAGS.num_gpus
            }
    main(**kwargs)


def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255

  return image, label



def main(epochs, buffer_size, batch_size, train_mode, distribution_strategy, num_gpus):
    #strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    #strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")


    strategy = get_distribution_strategy(strategy=distribution_strategy, num_gpus=num_gpus)
    print_msg ('Number of devices: {}'.format(strategy.num_replicas_in_sync), 'info')
   
    data_obj = Dataset(batch_size=128)
    train_dataset, test_dataset = data_obj.create_dataset()
    steps_per_epoch = data_obj.get_buffer_size()//(batch_size)
    train_obj = Benchmark(epochs, steps_per_epoch, 'resnet56')

    with strategy.scope():
        train_obj.create_model('resnet56')
        train_obj.compile_model()
        
    print_msg('Training...', 'info')
    train_obj.run(train_dataset, test_dataset, train_mode)
    print_msg('Training Done.', 'succ')

if __name__ == '__main__':
  app.run(run_main)
