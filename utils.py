import sys, os, json
import tensorflow as tf
from worker import *
from getpass import getpass


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


def get_distribution_strategy(strategy="OneDevice", num_gpus=0, workers=None, typ=None, index=None):
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
        if index == 0: setup_cluster(workers)
        
        os.environ['TF_CONFIG'] = json.dumps({
            'cluster': {
                'worker': workers.split(','),
            },
            'task': {
                'type': typ, 
                'index': index
            }
        })
        return tf.distribute.experimental.MultiWorkerMirroredStrategy()

def checkStatus(status, ret, exit=True):
    if status != 0:
        for line in ret:
            print_msg(line, 'err')
        if exit: sys.exit('Error!')

def setup_cluster(workers):
    print("############################################....Set up cluster...##########################################################")
    
    #remove the first element 'chief worker'
    hosts = workers.split(',')[1:]

    for host in hosts:
        print(host)
        host = host.split(':')
        host = host[0]
        port = host[1]

        user = input(f"Please enter Username for host {host}: ")
        pwd = getpass(f"Please enter Password for host {host}: ")
        
        config = Config(host, user, pwd, port=22)
        worker = Worker(config)
        worker.connect()

        #Clone repo in worker
        status, ret = worker.exec_cmd("git clone https://github.com/reger-men/tensorflow_benchmark.git ~/work/tensorflow_benchmark")
        checkStatus(status, ret, exit=False)

        #start training on worker
        status, ret = worker.exec_cmd("cd ~/work/tensorflow_benchmark")
        checkStatus(status, ret)

        status, ret = worker.exec_cmd("cd ~/work/tensorflow_benchmark && python3 train.py --train_mode='fit' --workers='192.168.1.183:122,192.168.1.185:123' --w_type='worker' --w_index=1 --distribution_strategy='MultiWorker'")
        checkStatus(status, ret)

