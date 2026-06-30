"""
TinyML-friendly block builders for ENAS v2.1.

Three block types are supported:
    1. standard_conv         : Conv → BN → Activation
    2. depthwise_separable   : DW → BN → Act → PW → BN → Act
    3. bottleneck            : Expand → DW → Project (MobileNetV2-style)

All blocks support optional residual skip connections when spatial
dimensions and channel counts align.
"""

import tensorflow as tf


def _get_activation(name):
    """Return Keras activation layer by name."""
    if name == "relu6":
        return tf.keras.layers.Activation(tf.nn.relu6)
    return tf.keras.layers.ReLU()


def standard_conv_block(x, filters, kernel_size, stride, activation, use_skip):
    """Standard Conv → BN → Activation with optional skip connection."""
    shortcut = x
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=stride,
        padding='same', use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = _get_activation(activation)(x)
    if use_skip and stride == 1 and shortcut.shape[-1] == filters:
        x = tf.keras.layers.Add()([x, shortcut])
    return x


def depthwise_separable_block(x, filters, kernel_size, stride,
                              activation, use_skip):
    """DW → BN → Act → PW → BN → Act with optional skip."""
    shortcut = x
    C_in     = x.shape[-1]

    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size, strides=stride,
        padding='same', use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = _get_activation(activation)(x)

    x = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = _get_activation(activation)(x)

    if use_skip and stride == 1 and C_in == filters:
        x = tf.keras.layers.Add()([x, shortcut])
    return x


def bottleneck_block(x, filters, kernel_size, stride, activation,
                     use_skip, expansion_ratio):
    """
    MobileNetV2-style inverted residual bottleneck.
    No activation after projection → quantisation-friendly.
    """
    shortcut = x
    C_in = x.shape[-1]
    mid  = max(1, int(C_in * expansion_ratio))

    if expansion_ratio > 1:
        x = tf.keras.layers.Conv2D(mid, 1, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = _get_activation(activation)(x)

    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size, strides=stride,
        padding='same', use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = _get_activation(activation)(x)

    x = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if use_skip and stride == 1 and C_in == filters:
        x = tf.keras.layers.Add()([x, shortcut])
    return x


def build_model(input_shape, num_classes, architecture):
    """
    Build a Keras model from a cell-based architecture dict.

    Parameters
    ----------
    input_shape : tuple (W, H, C)
    num_classes : int
    architecture : dict {'k': int, 'cells': [cell_dict, ...]}

    Returns
    -------
    (model, macc_estimate, cell_limited_bool)
    """
    import numpy as np

    k     = architecture["k"]
    cells = architecture["cells"]

    inputs = tf.keras.Input(shape=input_shape)
    n      = k

    # Stem
    x = tf.keras.layers.Conv2D(n, 3, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    multiplier         = 2.0
    cell_limited       = False

    for i, cell in enumerate(cells):
        if x.shape[1] <= 1 or x.shape[2] <= 1:
            cell_limited = True
            break

        n_out = int(np.ceil(n * multiplier))
        multiplier = multiplier - 2 ** -(i + 1)

        bt   = cell["block_type"]
        ks   = cell["kernel_size"]
        s    = cell["stride"]
        skip = cell["skip_connection"]
        act  = cell["activation"]
        er   = cell["expansion_ratio"]

        if bt == "standard_conv":
            x = standard_conv_block(x, n_out, ks, s, act, skip)
        elif bt == "depthwise_separable":
            x = depthwise_separable_block(x, n_out, ks, s, act, skip)
        elif bt == "bottleneck":
            x = bottleneck_block(x, n_out, ks, s, act, skip, er)

        n = n_out

    x       = tf.keras.layers.GlobalAveragePooling2D()(x)
    x       = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model   = tf.keras.Model(inputs=inputs, outputs=outputs)

    from enas.estimators import estimate_macc
    macc = estimate_macc(input_shape, architecture)
    return model, macc, cell_limited
