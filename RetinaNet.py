from ResNet101 import resnet_101
import tensorflow as tf

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
    new_size = [top.shape[0], top.shape[1] * 2, top.shape[2] * 2, top.shape[3]]
    input_top = tf.image.resize_nearest_neighbor(top, new_size)
    output = tf.add(input_left, input_top)
    return output


def focal_loss():
    pass


def class_subnet(input_layer, num_anchors, num_classes):
    layer = input_layer
    for _ in range(4):
        layer = tf.layers.conv2d(layer, filters=SUBNET_FILTERS, kernel_size=3)
        layer = tf.nn.relu(layer)

    final_layer_filters = num_anchors * num_classes
    output = tf.layers.conv2d(layer, filters=final_layer_filters, kernel_size=3, activation='sigmoid')
    return output


def box_subnet(input_layer, num_anchors):
    layer = input_layer
    for _ in range(4):
        layer = tf.layers.conv2d(layer, filters=SUBNET_FILTERS, kernel_size=3)
        layer = tf.nn.relu(layer)

    final_layer_filters = num_anchors * 4
    output = tf.layers.conv2d(layer, filters=final_layer_filters, kernel_size=3 )
    return output



def retina_subnets(fpn, num_anchors, num_classes):
    (p3, p4, p5, p6, p7) = fpn

    outputs = []

    for layer in fpn:
        prediction_class = class_subnet(layer, num_anchors, num_classes)
        prediction_box = box_subnet(layer, num_anchors)

        outputs.append()

    # TODO non-max suppresion with threshold of 0.5

    return outputs


def retinanet(inputs, targets):
    """
    Basic implementation of a RetinaNet with a ResNet101 backbone.
    Based on "Focal Loss for Dense Object Detection" (arXiv:1708.02002v2 [cs.CV] 7 Feb 2018)
    :param inputs: the input, e.g. images
    :param targets: the desired output, e.g. a class and maybe coordinates + anchor boxes for object detection
    :return:
    """
    with tf.name_scope("backbone"):
        with tf.name_scope("bottom-up"):
            residual_outputs, output_layer = resnet_101(inputs, targets)
            c2, c3, c4, c5 = residual_outputs

        with tf.name_scope('top-down'):
            p5 = tf.layers.conv2d(c5, filters=FEATURE_MAPS_FILTERS, kernel_size=1, strides=1)
            p6 = tf.layers.conv2d(c5, filters=FEATURE_MAPS_FILTERS, kernel_size=3, stride=2)

            p7 = tf.nn.relu(p6)
            p7 = tf.layers.conv2d(p7, kernel_size=3, strides=2)

            p4 = fpn_top_down_layer(p5, c4)
            p3 = fpn_top_down_layer(p4, c3)

            p2 = fpn_top_down_layer(p3, c2)  # TODO usually ignored, consider removing it

    with tf.name_scope("retina"):
        retina_subnets((p3, p4, p5, p6, p7))
