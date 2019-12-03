import os
import logging
import argparse
from pprint import pformat
import numpy as np
import scipy.stats
import tensorflow as tf
tf.enable_eager_execution()
from musegan.config import LOGLEVEL, LOG_FORMAT
from musegan.data import load_data, get_dataset, get_samples
from musegan.metrics import get_save_metric_ops
from musegan.model import Model
from musegan.utils import make_sure_path_exists, load_yaml
from musegan.utils import backup_src, update_not_none, setup_loggers

LOGGER = logging.getLogger("experimental")


def load_training_data(params, config):
    """Load and return the training data."""
    # Load data
    if params['is_conditional']:
        raise ValueError("Not supported yet.")
    else:
        labels = None
    LOGGER.info("Loading training data.")
    data = load_data(config['data_source'], config['data_filename'])
    LOGGER.info("Training data size: %d", len(data))
    
    # Build dataset
    LOGGER.info("Building dataset.")
    dataset = get_dataset(
        data, labels, config['batch_size'], params['data_shape'],
        config['use_random_transpose'], config['n_jobs'])
    
    # Create iterator
    if params['is_conditional']:
        train_x, train_y = dataset.make_one_shot_iterator().get_next()
    else:
        # train_x, train_y = dataset.make_one_shot_iterator().get_next(), None
        train_x, train_y = dataset, None
    return train_x, train_y

def process_maml_data():
    import time
    start_time = time.time()
    total_meta_batch = 3
    tasks_per_batch = 6
    classes_per_task = 4
    samples_per_class = 5
    train_per_class = 4
    val_per_class = 1
    num_total_classes = 30

    # Load data into separate training and validation iterators.
    data_path = '/home/canliu/aha/236/project/musegan/src/splits'
    data_source = 'npy'
    data_shape = [4, 48, 84, 5]
    train_Xs = []
    val_Xs = []
    for i in range(num_total_classes):
        # load train.
        data_filename = data_path + '/{}.npy'.format(i)
        data = load_data(data_source, data_filename)
        dataset = get_dataset(data, None, train_per_class, data_shape, False, 20)
        train_x = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
        train_Xs.append(train_x)
        # load validation.
        data_filename = data_path + '/{}.npy'.format(i * 2 + 1)
        data = load_data(data_source, data_filename)
        dataset = get_dataset(data, None, val_per_class, data_shape, False, 20)
        val_x = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
        val_Xs.append(val_x)
    
    ####### The rest is based on make_data_tensor() in maml's data generator #######
    # Sample tasks. 
    num_total_tasks = total_meta_batch * tasks_per_batch
    
    #### Dummy for sampler ####
    num_true = 1
    true_classes = tf.ones((1, 1), dtype=tf.dtypes.int64)
    ###########################
    

    print('total number of tasks: {}'.format(num_total_tasks))
    maml_train_x = []
    maml_val_x = []
    start = True
    config = tf.ConfigProto(device_count = {'GPU': 0})
    marker_1 = time.time()
    print('prepreprocess: {}'.format(marker_1 - start_time))
    with tf.Session(config=config) as sess:
        for i in range(num_total_tasks):
            iteration_start = time.time()
            print('iteration: {}'.format(i))
            sampled_classes = tf.random.uniform_candidate_sampler(true_classes, num_true, classes_per_task, unique=True, range_max=num_total_classes)
            sampled_classes = sampled_classes.sampled_candidates
            if start:
                print('sampled tasks: {}'.format(sampled_classes.eval()))
            iteration_sample = time.time()
            print('sample time: {}'.format(iteration_sample - iteration_start))
            # Sample examples for each class.
            samples = tf.concat([sess.run(train_Xs[cls]) for cls in sampled_classes.eval()], 0) # (classes * samples, 4, 48, 84, 5)
            samples_val = tf.concat([sess.run(val_Xs[cls]) for cls in sampled_classes.eval()], 0) # (classes * samples, 4, 48, 84, 5)
            if start:
                print('samples shape: {}'.format(samples.eval().shape))
            iteration_middle = time.time()
            print('load tensors: {}'.format(iteration_middle - iteration_sample))
            # Shuffle within task and reformat.
            processed_samples = []
            processed_samples_val = []
            idxs = tf.range(0, classes_per_task)
            for k in range(train_per_class):
                idxs = tf.random_shuffle(idxs)
                if start:
                    print('shuffled indices: {}'.format(idxs))
                #TODO: add labels.
                shifted_idxs = idxs * train_per_class + k
                processed_samples.append(tf.gather(samples, shifted_idxs))
            for k in range(val_per_class):
                idxs = tf.random_shuffle(idxs)
                shifted_idxs_val = idxs * val_per_class + k
                processed_samples_val.append(tf.gather(samples, shifted_idxs_val))
            processed_samples = tf.concat(processed_samples, 0)
            processed_samples_val = tf.concat(processed_samples_val, 0)
            if start:
                print('processed_samples shape: {}'.format(processed_samples.shape)) 
            start = False
            maml_train_x.append(processed_samples)
            maml_val_x.append(processed_samples_val)
            iteration_end = time.time()
            print('shift tensors: {}'.format(iteration_end - iteration_middle))
    print('length of maml_train_x: {}'.format(len(maml_train_x)))
    maml_train_x = tf.concat(maml_train_x, 0)
    maml_val_x = tf.concat(maml_val_x, 0)
    print('shape of concatenated maml train x: {}'.format(maml_train_x.shape))
    train_batch_size = tasks_per_batch * classes_per_task * train_per_class
    val_batch_size = tasks_per_batch * classes_per_task * val_per_class
    print('new batch size: {}'.format(train_batch_size))
    dataset_train_x = tf.data.Dataset.from_tensor_slices(maml_train_x).batch(train_batch_size)
    iterator_train = dataset_train_x.make_initializable_iterator()
    dataset_val_x = tf.data.Dataset.from_tensor_slices(maml_val_x).batch(val_batch_size)
    iterator_val = dataset_val_x.make_initializable_iterator()
    end_time = time.time()
    print('use time: {}'.format(end_time - start_time))
    return iterator_train.get_next(), iterator_val.get_next()

