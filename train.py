from __future__ import absolute_import, division, print_function, unicode_literals

from absl import app
from absl import flags
from utils import *

import tensorflow as tf
print_msg(f'Tensorflow version: {tf.__version__}', 'info')

# Filter out INFO & WARNING messages
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


from tensorflow.keras import datasets, layers, models
from benchmark import Benchmark
from data import Dataset


FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 1, 'Number of epochs')
flags.DEFINE_integer('buffer_size', 50000, 'Shuffle buffer size')
flags.DEFINE_integer('batch_size', 128, 'Batch Size')
flags.DEFINE_string('train_mode', 'fit', 'Use either keras fit or loop training')
flags.DEFINE_integer('display_every', 20, 'Number of steps after which progress is printed out')
flags.DEFINE_string('distribution_strategy', 'OneDevice', 'Can be: Mirrored, MultiWorker, OneDevice')
flags.DEFINE_integer('num_gpus', 1, 'Number of GPUs. 0 will run on CPU')
flags.DEFINE_string('workers', "localhost:122,localhost:123", 'List of workers IP:Port')
flags.DEFINE_string('w_type', "worker", 'Task type')
flags.DEFINE_integer('w_index', 0, 'Worker index. 0 is appointed as the chief worker')
flags.DEFINE_boolean('setup_cluster', True, 'Setup the cluster from the chief worker or not. This is an expiremental feature')
flags.DEFINE_integer('verbose', 0, 'Set verbosity level')

def run_main(argv):
    del argv
    kwargs = {
            'epochs': FLAGS.epochs,
            'buffer_size': FLAGS.buffer_size,
            'batch_size': FLAGS.batch_size,
            'train_mode': FLAGS.train_mode,
            'display_every': FLAGS.display_every,
            'distribution_strategy': FLAGS.distribution_strategy,
            'num_gpus': FLAGS.num_gpus,
            'workers': FLAGS.workers,
            'w_type': FLAGS.w_type,
            'w_index': FLAGS.w_index,
            'setup_cluster' : FLAGS.setup_cluster,
            'verbose': FLAGS.verbose,
            }
    main(**kwargs)


def main(epochs, buffer_size, batch_size, train_mode, display_every, 
        distribution_strategy, num_gpus,
        workers, w_type, w_index, setup_cluster, verbose):

    if verbose: os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(verbose)
    strategy = get_distribution_strategy(strategy=distribution_strategy, train_mode=train_mode, num_gpus=num_gpus, workers=workers, typ=w_type, index=w_index, setup=setup_cluster)
    if num_gpus == 1: num_gpus = strategy.num_replicas_in_sync
    print_msg ('Number of devices: {}'.format(num_gpus), 'info')
   
    data_obj = Dataset(batch_size)
    train_dataset, test_dataset = data_obj.create_dataset()
    steps_per_epoch = data_obj.get_buffer_size()//(batch_size)
    train_obj = Benchmark(epochs, steps_per_epoch, batch_size, display_every, num_gpus, 'resnet56', strategy)

    print_msg('Training...', 'info')
    train_obj.run(train_dataset, test_dataset, train_mode)
    print_msg('Training Done.', 'succ')

if __name__ == '__main__':
    app.run(run_main)
