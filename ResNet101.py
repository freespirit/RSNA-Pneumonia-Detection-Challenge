import tensorflow as tf


def make_3layer_name(base):
    return f"{base}a", f"{base}b", f"{base}c"


def normalized_conv2d(inputs, filters, kernel_size, strides, padding='valid', name=""):
    conv = tf.layers.conv2d(inputs, filters, kernel_size, strides, padding, name=name)
    norm = tf.layers.batch_normalization(conv, name=f"{name}_bn")
    return norm


def normalized_conv2d_with_relu(inputs, filters, kernel_size, strides, padding='valid', name=""):
    conv = tf.layers.conv2d(inputs, filters, kernel_size, strides, padding, name=name)
    norm = tf.layers.batch_normalization(conv, name=f"{name}_bn")
    relu = tf.nn.relu(norm, name=f"{name}_relu")
    return relu


def residual_identity_block(inputs, filters, base_layer_name):
    """
    A residual block with identity shortcut. This block simply connects (via an addition operation) the input with the
    result of the internal convolutional layers
    """
    shortcut = inputs

    l1, l2, l3 = make_3layer_name(base_layer_name)

    prev = normalized_conv2d_with_relu(inputs, filters[0], kernel_size=1, strides=1, padding='valid', name=l1)
    prev = normalized_conv2d_with_relu(prev, filters[1], kernel_size=3, strides=1, padding='same', name=l2)
    prev = normalized_conv2d(prev, filters[2], kernel_size=1, strides=1, padding='valid', name=l3)

    output = tf.add(shortcut, prev, name=f"{base_layer_name}_shortcut")
    output = tf.nn.relu(output, name=f"{base_layer_name}_relu")

    return output


def residual_conv_block(inputs, filters, base_layer_name, s=2):
    """
    A residual block with convolutions shortcut. Similar to the identity block except for applying convolution on the shortcut
    """
    shortcut_name = f"{base_layer_name}_shortcut"
    shortcut = tf.layers.conv2d(inputs, filters[2], kernel_size=1, strides=s, padding='valid', name=shortcut_name)
    shortcut = tf.layers.batch_normalization(shortcut, axis=3, name=f"{shortcut_name}_bn")

    l1, l2, l3 = make_3layer_name(base_layer_name)

    prev = normalized_conv2d_with_relu(inputs, filters[0], kernel_size=1, strides=s, padding='valid', name=l1)
    prev = normalized_conv2d_with_relu(prev, filters[1], kernel_size=3, strides=1, padding='same', name=l2)
    prev = normalized_conv2d(prev, filters[2], kernel_size=1, strides=1, padding='valid', name=l3)

    output = tf.add(shortcut, prev, name=f"{base_layer_name}_add")
    output = tf.nn.relu(output, name=f"{base_layer_name}_relu")

    return output


def residual_block(inputs, filters, blocks, base_layer_name):
    with tf.name_scope(base_layer_name):
        prev = residual_conv_block(inputs, filters, f"{base_layer_name}_1", s=2)
        for i in range(1, blocks):
            prev = residual_identity_block(prev, filters=filters, base_layer_name=f"{base_layer_name}_{i+1}")

    return prev


def resnet_101(inputs, classes):
    with tf.name_scope("common"):
        c1 = tf.layers.conv2d(inputs, kernel_size=7, strides=2, filters=64, padding="same", name='conv1')
        c1 = tf.layers.batch_normalization(c1, name="conv1_bn")
        c1 = tf.nn.relu(c1, name="conv1_relu")
        c1 = tf.layers.max_pooling2d(c1, pool_size=3, strides=2)

    with tf.name_scope("residual"):
        c2 = residual_block(c1, base_layer_name="conv2", blocks=3, filters=[64, 64, 256])
        c3 = residual_block(c2, base_layer_name="conv3", blocks=4, filters=[128, 128, 512])
        c4 = residual_block(c3, base_layer_name="conv4", blocks=23, filters=[256, 256, 1024])
        c5 = residual_block(c4, base_layer_name="conv5", blocks=3, filters=[512, 512, 2048])

    with tf.name_scope("common"):
        avg = tf.layers.average_pooling2d(inputs=c5, pool_size=2, strides=1)
        flat = tf.layers.flatten(avg, name="flat")
        dense = tf.layers.dense(inputs=flat, units=classes, name='fc')

    return (c2, c3, c4, c5), dense
