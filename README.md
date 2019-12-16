# Tensorflow v2 benchmark
TensorFlow benchmark scripts for single and multi Nodes with Multi GPUs

#### Usage
##### Clone repo
```git clone https://github.com/reger-men/tensorflow_benchmark.git```

##### Pre-requirement
```pip3 install -r requirements.txt```

##### Single Node Single GPU
Train with custom loop:

```python3 train.py -train-mode loop```

Train with Keras fit:

```python3 train.py -train-mode fit```

##### Single Node Multi-GPUs
Mirrored strategy will be used as defalt with ```num_gpus>1```

```python3 train.py -train_mode fit -num_gpus 2```

Loop still under development.

##### Multi-Node Multi-GPUs
Experimental: launch Multi-Nodes training from the chief Worker Node

```python3 train.py --train_mode=fit --workers="localhost:122,localhost:123" --w_type="worker" --w_index=0 --distribution_strategy=MultiWorker```

Loop still under development.
