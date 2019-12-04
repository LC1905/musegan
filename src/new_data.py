"""This file contains functions for loading and preprocessing pianoroll data.
"""
import logging
import numpy as np
import tensorflow as tf
from musegan.config import SHUFFLE_BUFFER_SIZE, PREFETCH_SIZE
LOGGER = logging.getLogger(__name__)

# --- Data loader --------------------------------------------------------------
def load_data_from_npy(filename):
    """Load and return the training data from a npy file."""
    return np.load(filename)

def load_data_from_npz(filename):
    """Load and return the training data from a npz file (sparse format)."""
    with np.load(filename) as f:
        data = np.zeros(f['shape'], np.bool_)
        data[[x for x in f['nonzero']]] = True
    return data

def load_data(data_source, data_filename):
    """Load and return the training data."""
    if data_source == 'sa':
        import SharedArray as sa
        return sa.attach(data_filename)
    if data_source == 'npy':
        return load_data_from_npy(data_filename)
    if data_source == 'npz':
        return load_data_from_npz(data_filename)
    raise ValueError("Expect `data_source` to be one of 'sa', 'npy', 'npz'. "
                     "But get " + str(data_source))

# --- Dataset Utilities -------------------------------------------------------
def random_transpose(pianoroll):
    """Randomly transpose a pianoroll with [-5, 6] semitones."""
    semitone = np.random.randint(-5, 6)
    if semitone > 0:
        pianoroll[:, semitone:, 1:] = pianoroll[:, :-semitone, 1:]
        pianoroll[:, :semitone, 1:] = 0
    elif semitone < 0:
        pianoroll[:, :semitone, 1:] = pianoroll[:, -semitone:, 1:]
        pianoroll[:, semitone:, 1:] = 0
    return pianoroll

def set_pianoroll_shape(pianoroll, data_shape):
    """Set the pianoroll shape and return the pianoroll."""
    pianoroll.set_shape(data_shape)
    return pianoroll

def set_label_shape(label):
    """Set the label shape and return the label."""
    label.set_shape([1])
    return label

# --- Sampler ------------------------------------------------------------------
def get_samples(n_samples, data, labels=None, use_random_transpose=False):
    """Return some random samples of the training data."""
    indices = np.random.choice(len(data), n_samples, False)
    if np.issubdtype(data.dtype, np.bool_):
        sample_data = data[indices] * 2. - 1.
    else:
        sample_data = data[indices]
    if use_random_transpose:
        sample_data = np.array([random_transpose(x) for x in sample_data])
    if labels is None:
        return sample_data
    return sample_data, labels[indices]


def sample( batch_num=3, task_per_batch=6, class_per_task=4, sam_per_class=4, val_per_class=2, NUM_OF_CLASS =25,labels =None):
    train_class = [20, 23, 14, 5, 12, 19, 17, 8, 0, 13, 16, 1, 3, 9, 4]
    train_class_num = 15
    # train_filename = [ str(i)+"_train" for i in range(25)]
    # test_filename  = [ str(i)+"_test" for i in range(25)]
    train_filename = [ str(i)+"_train" for i in train_class]
    test_filename  = [ str(i)+"_test" for i in train_class]
    train_data_list = []
    valid_data_list = []
    import SharedArray as sa
    # for i in range(NUM_OF_CLASS):
    for i in range(train_class_num):
        train_data_list.append(sa.attach(train_filename[i]))
        valid_data_list.append(sa.attach(test_filename[i]))
    train_batch = []
    valid_batch = []
    for i in range(batch_num):
        tasks_train = []
        tasks_valid = []
        for j in range(task_per_batch):
            t = numpy.random.choice(len(train_class),class_per_task,replace=False)
            # t = numpy.random.choice(range(NUM_OF_CLASS), class_per_task, replace=False)
            train_temp = []
            valid_temp = []
            for k in t:
                train_temp.append(train_data_list[k][numpy.random.choice(len(train_data_list[k]),sam_per_class,replace=False)])
                valid_temp.append(valid_data_list[k][numpy.random.choice(len(valid_data_list[k]),val_per_class,replace=False)])
                # shape=(sam/val_per_class,4,48,84,5)
                # shape=(cls_per_task,sam/val_per_class,4,48,84,5)
            train_temp = np.transpose(train_temp,(1,0,2,3,4,5))
            train_temp = np.expand_dims(train_temp,0)
            valid_temp = np.transpose(valid_temp,(1,0,2,3,4,5))
            valid_temp = np.expand_dims(valid_temp,0)
            tasks_train.append(train_temp)
            tasks_valid.append(valid_temp)
        train_batch.append(np.expand_dims(np.concatenate(tasks_train,0),0)) # shape = (task_per_bath,cls_per_task,sam/val_per_class,4,48,84,5)
        valid_batch.append(np.expand_dims(np.concatenate(tasks_valid,0),0)) # the same
    train_x = np.concatenate(train_batch,0)
    valid_x = np.concatenate(valid_batch,0)
    print(train_x.shape)
    print(valid_x.shape)
    return train_x, valid_x


# --- Tensorflow Dataset -------------------------------------------------------
def _gen_data(data, labels=None):
    """Data Generator."""
    if labels is None:
        for item in data:
            if np.issubdtype(data.dtype, np.bool_):
                yield item * 2. - 1.
            else:
                yield item
    else:
        for i, item in enumerate(data):
            if np.issubdtype(data.dtype, np.bool_):
                yield (item * 2. - 1., labels[i])
            else:
                yield (item, labels[i])

def get_dataset(data, labels=None, batch_size=None, data_shape=None,
                use_random_transpose=False, num_threads=1):
    """Create  and return a tensorflow dataset from an array."""
    if labels is None:
        dataset = tf.data.Dataset.from_generator(
            lambda: _gen_data(data), tf.float32)
        if use_random_transpose:
            dataset = dataset.map(
                lambda pianoroll: tf.py_func(
                    random_transpose, [pianoroll], tf.float32),
                num_parallel_calls=num_threads)
        dataset = dataset.map(lambda pianoroll: set_pianoroll_shape(
            pianoroll, data_shape), num_parallel_calls=num_threads)
    else:
        assert len(data) == len(labels), (
            "Lengths of `data` and `lables` do not match.")
        dataset = tf.data.Dataset.from_generator(
            lambda: _gen_data(data, labels), [tf.float32, tf.int32])
        if use_random_transpose:
            dataset = dataset.map(
                lambda pianoroll, label: (
                    tf.py_func(random_transpose, [pianoroll], tf.float32),
                    label),
                num_parallel_calls=num_threads)
        dataset = dataset.map(
            lambda pianoroll, label: (set_pianoroll_shape(
                pianoroll, data_shape), set_label_shape(label)),
            num_parallel_calls=num_threads)

    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).repeat().batch(batch_size)
    return dataset.prefetch(PREFETCH_SIZE)
