"""Tensorflow ops."""
import tensorflow as tf

CONV_KERNEL_INITIALIZER = tf.truncated_normal_initializer(stddev=0.05)
DENSE_KERNEL_INITIALIZER = tf.truncated_normal_initializer(stddev=0.05)

dense = lambda i, u: tf.layers.dense(
    i, u, kernel_initializer=DENSE_KERNEL_INITIALIZER)

conv2d = lambda i, f, k, s: tf.layers.conv2d(
    i, f, k, s, kernel_initializer=CONV_KERNEL_INITIALIZER)

#### conv3d with layer and nn version. 
conv3d = lambda i, f, k, s: tf.layers.conv3d(
    i, f, k, s, kernel_initializer=CONV_KERNEL_INITIALIZER)
nn_conv3d = lambda i, f, k, s: tf.nn.conv3d(
    input=i, filter=f, strides=s, padding='VALID') # ignore k 
"""
tf.layers.conv3d(
    inputs,
    filters,
    kernel_size,
    strides=(1, 1, 1),
    padding='valid',
    data_format='channels_last',
    dilation_rate=(1, 1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    reuse=None
)
tf.nn.conv3d(
    input,
    filter,
    strides,
    padding,
    data_format='NDHWC',
    dilations=[1, 1, 1, 1, 1],
    name=None
)
"""


tconv2d = lambda i, f, k, s: tf.layers.conv2d_transpose(
    i, f, k, s, kernel_initializer=CONV_KERNEL_INITIALIZER)

# tconv3d with nn and layer version 
tconv3d = lambda i, f, k, s: tf.layers.conv3d_transpose(
    i, f, k, s, kernel_initializer=CONV_KERNEL_INITIALIZER)
nn_tconv3d = lambda i, f, o, s: tf.nn.conv3d_transpose(
    i, f, o, s)

# https://zhuanlan.zhihu.com/p/48501100 formula

"""
tf.layers.conv3d_transpose(
    inputs,
    filters,
    kernel_size,
    strides=(1, 1, 1),
    padding='valid',
    data_format='channels_last',
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    reuse=None
)
tf.nn.conv3d_transpose(
    value,
    filter,
    output_shape,
    strides,
    padding='SAME',
    data_format='NDHWC',
    name=None
)
"""


def get_normalization(norm_type, training=None):
    """Return the normalization function."""
    if norm_type == 'batch_norm':
        return lambda x: tf.layers.batch_normalization(x, training=training)
    if norm_type == 'layer_norm':
        return tf.contrib.layers.layer_norm
    if norm_type is None or norm_type == '':
        return lambda x: x
    raise ValueError("Unrecognizable normalization type.")
