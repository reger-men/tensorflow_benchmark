# Tensorflow v2 benchmark
![Supports TFv2](https://img.shields.io/badge/Supports-tensorflow%20v2-blue.svg)

TensorFlow benchmark scripts for single and multi Nodes with Multi GPUs

#### Usage
##### Clone repo
```git clone https://github.com/reger-men/tensorflow_benchmark.git```

##### Pre-requirement
```pip3 install -r requirements.txt```

##### Single Node Single GPU
Train with custom loop:

```python3 train.py -train_mode loop```

Train with Keras fit:

```python3 train.py -train_mode fit```

##### Single Node Multi-GPUs
Mirrored strategy will be used as default with ```num_gpus>1```

```python3 train.py -train_mode fit -num_gpus 2```

```python3 train.py -train_mode loop -num_gpus 2```

##### Multi-Node Multi-GPUs
Experimental: launch Multi-Nodes training from the chief Worker Node

```python3 train.py --train_mode=fit --workers="localhost:122,localhost:123" --w_type="worker" --w_index=0 --distribution_strategy=MultiWorker```

Loop still under development.

##### Flags
```python3 train.py --helpfull```

```
train.py:
  --batch_size: Batch Size
    (default: '128')
    (an integer)
  --buffer_size: Shuffle buffer size
    (default: '50000')
    (an integer)
  --display_every: Number of steps after which progress is printed out
    (default: '20')
    (an integer)
  --distribution_strategy: Can be: Mirrored, MultiWorker, OneDevice
    (default: 'OneDevice')
  --epochs: Number of epochs
    (default: '1')
    (an integer)
  --num_gpus: Number of GPUs. 0 will run on CPU
    (default: '1')
    (an integer)
  --[no]setup_cluster: Setup the cluster from the chief worker or not. This is an expiremental feature
    (default: 'true')
  --train_mode: Use either keras fit or loop training
    (default: 'fit')
  --verbose: Set verbosity level
    (default: '0')
    (an integer)
  --w_index: Worker index. 0 is appointed as the chief worker
    (default: '0')
    (an integer)
  --w_type: Task type
    (default: 'worker')
  --workers: List of workers IP:Port
    (default: 'localhost:122,localhost:123')
  ```
