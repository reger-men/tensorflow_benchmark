import sys
import tensorflow as tf


def print_msg(msg, typ=None, onLine=False):
    """Print msg in specific format by it type"""
    
    end_ = '\n'
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
    
    if onLine:
        end_ = '\r'

    print(msg, end=end_)


def get_distribution_strategy(strategy="OneDevice", num_gpus=0):
    if num_gpus == 0:
        devices = ["device:CPU:0"]
    elif strategy == "OneDevice" and num_gpus > 1:
        strategy = "Mirrored"

    if strategy == "OneDevice":
        if num_gpus > 0:
            devices = ["device:GPU:{}".format(num_gpus-1)]
        return tf.distribute.OneDeviceStrategy(devices[0])

    elif strategy == "Mirrored":
        if num_gpus > 0:
            devices = ["device:GPU:%d" % i for i in range(num_gpus)]
        return tf.distribute.MirroredStrategy(devices=devices, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    elif strategy == "MultiWorker":
        return tf.distribute.experimental.MultiWorkerMirroredStrategy()

