from __future__ import absolute_import, division, print_function, unicode_literals
from utils import *
import tensorflow as tf
from models import resnet_cifar_model
import time
import numpy as np

'''tensorflow/benchmarks/blob/2696206fc01860d7b06fd02a01626f56abbab40a/scripts/tf_cnn_benchmarks/benchmark_cnn.py#L934'''
def get_perf_timing(batch_size, step_train_times, scale=1):
    """Calculate benchmark processing speed."""
    times = np.array(step_train_times)
    speeds = batch_size / times
    time_mean = np.mean(times)
    speed_mean = scale * batch_size / time_mean
    speed_uncertainty = np.std(speeds) / np.sqrt(float(len(speeds)))
    speed_jitter = 1.4826 * np.median(np.abs(speeds - np.median(speeds)))
    return speed_mean, speed_uncertainty, speed_jitter

class Callbacks(tf.keras.callbacks.Callback):

    def __init__(self, batch_size, display_every=20, num_gpu=1):
        self.batch_size = batch_size
        self.display_every = display_every
        self.num_gpu = num_gpu
        self.start_time = 0
        self.train_time = 0
        self.step_train_times = []
        self.speeds = []
        self._chief_worker_only = True

    def on_train_batch_begin(self, batch, logs=None):
        if (batch > 0):
                self.start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if (batch > 0):
            self.train_time = time.time() - self.start_time
            self.step_train_times.append(self.train_time)

            if ((batch % self.display_every) == 0):
                speed_mean, speed_uncertainty, speed_jitter = get_perf_timing(self.batch_size, self.step_train_times, self.num_gpu)
                self.speeds.append(speed_mean)

                log_str = "{d0:d}\t{f1:0.1f}\t\t{f2:0.4f}\t\t{f3:0.4f}".format(d0=batch, f1=speed_mean, f2=logs['loss'], f3=logs['accuracy']*100)
                print_msg(log_str, 'info')

    def on_train_end(self, logs=None):
        if (self.speeds):
            speeds = np.array(self.speeds)
            speed_mean = np.mean(speeds)
            log_str = "total images/sec: {f0:0.2f}".format(f0=speed_mean)
            print_msg(log_str, 'step')


class Benchmark(object):
    def __init__(self, epochs, steps_per_epoch, batch_size=128, display_every=20, num_gpu=1, model='resnet56', strategy=None):
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.display_every = display_every
        self.num_gpu = num_gpu
        self.strategy = strategy

        with self.strategy.scope():
            self.model = self.create_model(model_name=model)
            self.loss_fit = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            self.loss_loop = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True)
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1,momentum=0.9, nesterov=True)
            checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        self.callbacks = [Callbacks(self.batch_size, self.display_every, self.num_gpu), tf.keras.callbacks.LearningRateScheduler(self.decay)]

    def create_model(self, model_name):
        if model_name == 'resnet56':
            self.model = resnet_cifar_model.resnet56(classes=10)
        else:
            err_msg = "Model name \"{}\" cannot be found!".format(model_name)
            print_msg(err_msg, 'err')
            sys.exit('Error!')
        return self.model

    def compile_model(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fit, metrics=['accuracy'])

    def decay(self, epoch):
      if epoch < 3:
        return 1e-3
      elif epoch >= 3 and epoch < 7:
        return 1e-4
      else:
        return 1e-5

    def compute_loss(self, labels, predictions, batch_size=128):
        per_example_loss = self.loss_loop(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=batch_size)

    def train_step(self, inputs):
        image, label = inputs
        with tf.GradientTape() as tape:
            predictions = self.model(image, training=True)
            loss = self.compute_loss(label, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(label, predictions)
        return loss

    @tf.function
    def distributed_train_step(self, dataset_inputs):
        per_replica_losses = self.strategy.experimental_run_v2(self.train_step, args=(dataset_inputs,))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    
    def test_step(inputs):
        images, labels = inputs

        predictions = self.model(images, training=False)
        t_loss = self.loss_loop(labels, predictions)

        test_loss.update_state(t_loss)
        test_accuracy.update_state(labels, predictions)
    
    def run(self, train_dataset, test_dataset, train_mode):
        if train_mode == 'loop':
            return self.loop_train(train_dataset, test_dataset)
        elif train_mode == 'fit':
            return self.fit_train(train_dataset.repeat(), test_dataset.repeat())

    def loop_train(self, train_dataset, test_dataset):
        #convert a tf.data.Dataset to "per-replica" values
        if self.strategy._tf_api_names[0] != "distribute.OneDeviceStrategy": 
            train_dist_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
            test_dist_dataset = self.strategy.experimental_distribute_dataset(test_dataset)
        else:
            train_dist_dataset = train_dataset
            test_dist_dataset = test_dataset

        with self.strategy.scope():
            print_msg("Warming Up...", 'info')
            for x in train_dataset.take(self.num_gpu):
                self.distributed_train_step(x)

            header_str = "{s0:s}\t{s1:s}\t\t{s2:s}\t{s3:s}".format(s0='Step', s1='Img/sec', s2='total_loss', s3='accuracy')
            print_msg(header_str, 'step')
        
            batch=0
            step_train_times = []
            speeds = []

            for epoch in range(self.epochs):
                for inputs in train_dist_dataset:
                    # store start time
                    start_time = time.time()

                    self.distributed_train_step(inputs)

                    template = ('Epoch: {}, Train Loss: {}, Train Accuracy: {}, '
                                  'Test Loss: {}, Test Accuracy: {}')
                    batch += 1
                    if ((batch % self.display_every) == 0):
                        #Measuring elapsed time
                        train_time = time.time() - start_time
                        step_train_times.append(train_time)

                        speed_mean, speed_uncertainty, speed_jitter = get_perf_timing(self.batch_size, step_train_times, self.num_gpu)
                        speeds.append(speed_mean)

                        log_str = "{d0:d}\t{f1:0.1f}\t\t{f2:0.4f}\t\t{f3:0.4f}".format(d0=batch, f1=speed_mean, f2=self.train_loss.result(), f3=self.train_accuracy.result())
                        print_msg(log_str, 'info')
            
            speeds = np.array(speeds)
            speed_mean = np.mean(speeds)
            log_str = "total images/sec: {f0:0.2f}".format(f0=speed_mean)
            print_msg(log_str, 'step')


    def fit_train(self, train_dataset, test_dataset):
        # Compile the model in the context of strategy.scope
        with self.strategy.scope():
            self.compile_model()

        print_msg("Warming Up...", 'info')
        self.model.fit(train_dataset.take(self.num_gpu), epochs=1, steps_per_epoch=1, verbose=0)

        header_str = "{s0:s}\t{s1:s}\t\t{s2:s}\t{s3:s}".format(s0='Step', s1='Img/sec', s2='total_loss', s3='accuracy')
        print_msg(header_str, 'step')

        history = self.model.fit(train_dataset, epochs=self.epochs, steps_per_epoch=self.steps_per_epoch, verbose=0, callbacks=self.callbacks)
        return (history.history['loss'][-1],
                history.history['accuracy'][-1])

