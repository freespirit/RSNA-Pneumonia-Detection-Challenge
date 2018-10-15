import math

import tensorflow as tf

from ResNet101 import resnet_101

FEATURE_MAPS_FILTERS = 256
SUBNET_FILTERS = 256


def fpn_top_down_layer(top, left):
    """
    A layer in the top-down branch of a Feature Pyramid Network.
    Based on "Feature Pyramid Networks for Object Detection" (arXiv:1612.03144v2 [cs.CV] 19 Apr 2017)
    :param top: the layer above in the top-down branch of the FPN
    :param left: the layer from the left branch, e.g. a residual block from a backbone ResNet
    :return:
    """

    input_left = tf.layers.conv2d(left, filters=FEATURE_MAPS_FILTERS, kernel_size=1, strides=1)
    new_size = [top.shape[1] * 2, top.shape[2] * 2]
    input_top = tf.image.resize_nearest_neighbor(top, new_size)
    output = tf.add(input_left, input_top)
    return output


def class_subnet(input_layer, num_anchors, num_classes, prior=0.01):
    bias_initializer = tf.zeros_initializer()
    kernel_intializer = tf.random_normal_initializer(stddev=0.01)
    layer = input_layer
    for _ in range(4):
        layer = tf.layers.conv2d(layer, filters=SUBNET_FILTERS,
                                 kernel_size=3,
                                 padding='same',
                                 bias_initializer=bias_initializer,
                                 kernel_initializer=kernel_intializer)
        layer = tf.nn.relu(layer)

    initial_bias = - math.log((1 - prior) / prior)
    bias_initializer = tf.constant_initializer(initial_bias)
    final_layer_filters = num_anchors * num_classes
    output = tf.layers.conv2d(layer, filters=final_layer_filters,
                              kernel_size=3,
                              padding='same',
                              bias_initializer=bias_initializer,
                              kernel_initializer=kernel_intializer)
    output = tf.nn.sigmoid(output)
    return tf.layers.flatten(output)


def box_subnet(input_layer, num_anchors):
    bias_initializer = tf.zeros_initializer()
    kernel_initializer = tf.random_normal_initializer(stddev=0.01)
    layer = input_layer
    for _ in range(4):
        layer = tf.layers.conv2d(layer, filters=SUBNET_FILTERS,
                                 kernel_size=3,
                                 padding='same',
                                 bias_initializer=bias_initializer,
                                 kernel_initializer=kernel_initializer)
        layer = tf.nn.relu(layer)

    final_layer_filters = num_anchors * 4
    output = tf.layers.conv2d(layer, filters=final_layer_filters,
                              kernel_size=3,
                              padding='same',
                              bias_initializer=bias_initializer,
                              kernel_initializer=kernel_initializer)
    output = tf.nn.sigmoid(output)
    return tf.layers.flatten(output)


def retina_subnets(fpn, num_anchors, num_classes):
    (p3, p4, p5, p6, p7) = fpn

    # TODO Anchors! Section 4
    # This is importan! See "5.2. Model Architecture Design"
    # Experiment with Aspect Ratios and sizes

    outputs = []

    for layer in fpn:
        class_logits = class_subnet(layer, num_anchors, num_classes)
        box_logits = box_subnet(layer, num_anchors)

        outputs.append((class_logits, box_logits))

    # TODO non-max suppresion with threshold of 0.5

    return outputs

# TODO weight decay of 0.0001 and a momentum of 0.9


def retinanet(inputs, target_classes, prior=0.01):
    """
    Basic implementation of a RetinaNet with a ResNet101 backbone.
    Based on "Focal Loss for Dense Object Detection" (arXiv:1708.02002v2 [cs.CV] 7 Feb 2018)
    :param inputs: the input, e.g. images
    :param target_classes: the number of target classes
    :param prior: a constant used to counter class imbalance. Used in model initialization.
     See "3.3. Class Imbalance and Model Initialization" in the RetinaNet paper
    :return:
    """
    with tf.name_scope("backbone"):
        with tf.name_scope("bottom-up"):
            residual_outputs, output_layer = resnet_101(inputs, target_classes)
            c2, c3, c4, c5 = residual_outputs

        with tf.name_scope('top-down'):
            initializer = tf.initializers.random_normal(stddev=0.01)
            p5 = tf.layers.conv2d(c5, FEATURE_MAPS_FILTERS, kernel_size=1, strides=1)
            p6 = tf.layers.conv2d(c5, FEATURE_MAPS_FILTERS, kernel_size=3, strides=2, kernel_initializer=initializer)

            p7 = tf.nn.relu(p6)
            p7 = tf.layers.conv2d(p7, FEATURE_MAPS_FILTERS, kernel_size=3, strides=2, kernel_initializer=initializer)

            p4 = fpn_top_down_layer(p5, c4)
            p3 = fpn_top_down_layer(p4, c3)
            # p2 = fpn_top_down_layer(p3, c2)  Not actually used in the RetinNet original architecture

    with tf.name_scope("retina"):
        return retina_subnets((p3, p4, p5, p6, p7), 9, 1) ## TODO parameters


def focal_loss(logits, target, alpha, gamma, normalizer):
    """
    Focal loss function - extension of Cross Entropy as described in "Focal Loss for Dense Object Detection"
    (arXiv:1708.02002v2 [cs.CV] 7 Feb 2018)
    :param alpha:
    :param gamma:
    :param normalizer:
    :return:
    """
    # FL(pt) = − α()(1 − pt)**γ) * log(pt) #. (4)
    with tf.name_scope('focal_loss'):
        positive_label_mask = tf.equal(target, 1)
        cross_entropy = tf.losses.sigmoid_cross_entropy(target, logits)
        # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits)
        probs = tf.sigmoid(logits)
        probs_gt = tf.where(positive_label_mask, probs, 1.0 - probs)
        modulator = tf.pow(1.0 - probs_gt, gamma)
        loss = modulator * cross_entropy
        weighted_loss = tf.where(positive_label_mask, alpha * loss, (1.0 - alpha) * loss)
        total_loss = tf.reduce_sum(weighted_loss)
        total_loss /= normalizer
    return total_loss


def class_loss(class_outputs, target_class, num_positives, alpha=0.5, gamma=2.0):
    normalizer = num_positives
    classification_loss = focal_loss(class_outputs, target_class, alpha, gamma, normalizer)
    return classification_loss


def bbox_loss(box_outputs, box_targets, num_positives, delta=0.1):
    normalizer = num_positives * 4.0
    mask = tf.not_equal(box_targets, 0.0)
    box_loss = tf.losses.huber_loss(box_targets, box_outputs, weights=mask, delta=delta, reduction=tf.losses.Reduction.SUM)
    box_loss /= normalizer
    return box_loss


def retina_loss(outputs, target_class, target_box, num_classes=1):
    class_losses = []
    box_losses = []

    target_class = tf.one_hot(target_class, num_classes)
    target_class = tf.reshape(target_class, [-1, num_classes])

    for predictions in outputs:
        class_outputs, box_outputs = predictions

        target = tf.broadcast_to(target_class, tf.shape(class_outputs))
        # print(target)
        class_losses.append(class_loss(class_outputs, target, 1))

        target = tf.broadcast_to(target_box, tf.shape(box_outputs))
        # print(target)
        box_losses.append(bbox_loss(box_outputs, target, 1))

    return tf.add_n(class_losses) + tf.add_n(box_losses)
