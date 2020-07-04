from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras.applications import imagenet_utils
from CONSTANTS import NUM_CLASSES, NUM_PREDS_PER_BOX


def custom_inv_res_block(inputs, expansion, stride, alpha, filters, block_id):
    """Inverted ResNet block."""
    channel_axis = -1

    in_channels = backend.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        out = layers.Conv2D(
                expansion * in_channels,
                kernel_size=1,
                padding='same',
                use_bias=False,
                activation=None,
                name=prefix + 'expand')(
                        x)
        x = layers.BatchNormalization(
                axis=channel_axis,
                epsilon=1e-3,
                momentum=0.999,
                name=prefix + 'expand_BN')(
                        out)
        x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = layers.ZeroPadding2D(
                padding=correct_pad(x, 3),
                name=prefix + 'pad')(x)
    x = layers.DepthwiseConv2D(
            kernel_size=3,
            strides=stride,
            activation=None,
            use_bias=False,
            padding='same' if stride == 1 else 'valid',
            name=prefix + 'depthwise')(
                    x)
    x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'depthwise_BN')(
                    x)

    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(
            pointwise_filters,
            kernel_size=1,
            padding='same',
            use_bias=False,
            activation=None,
            name=prefix + 'project')(
                    x)
    x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'project_BN')(
                    x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x, out


#    functions used for generating mobilenets intermediate blocks
def inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    """Inverted ResNet block."""
    channel_axis = -1

    in_channels = backend.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = layers.Conv2D(
                expansion * in_channels,
                kernel_size=1,
                padding='same',
                use_bias=False,
                activation=None,
                name=prefix + 'expand')(
                        x)
        x = layers.BatchNormalization(
                axis=channel_axis,
                epsilon=1e-3,
                momentum=0.999,
                name=prefix + 'expand_BN')(
                        x)
        x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = layers.ZeroPadding2D(
                padding=correct_pad(x, 3),
                name=prefix + 'pad')(x)
    x = layers.DepthwiseConv2D(
            kernel_size=3,
            strides=stride,
            activation=None,
            use_bias=False,
            padding='same' if stride == 1 else 'valid',
            name=prefix + 'depthwise')(
                    x)
    x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'depthwise_BN')(
                    x)

    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(
            pointwise_filters,
            kernel_size=1,
            padding='same',
            use_bias=False,
            activation=None,
            name=prefix + 'project')(
                    x)
    x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'project_BN')(
                    x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x


def Relu6_Conv(x,filters,id):
    ''' 
    performs relu6 conv according to ssdlite paper
    '''
    pred = layers.Conv2D(filters, kernel_size=1, strides=(2,2),\
        use_bias=False, name= str(id) + '_pred_cnv')(x)
    
    x = layers.BatchNormalization( axis=-1, epsilon=1e-3, \
        momentum=0.999, name= str(id) + '_pred_bn')(pred)
    
    x = layers.ReLU(6., name= str(id) + '_expand_relu')(x)

    return x, pred


def ssd_head(x,filters,id,depthwise_stride=2):

    x = layers.Conv2D(filters//2,kernel_size=1,padding='same', use_bias=False, \
        name= str(id) + '_ssd_head_cnv')(x)
    
    x = layers.BatchNormalization( axis=-1, epsilon=1e-3, momentum=0.999, \
        name= str(id) + '_ssd_head_bn')(x)

    x = layers.DepthwiseConv2D( kernel_size=3, strides=depthwise_stride, \
        activation=None, use_bias=False, padding='same', \
            name= str(id) + 'depthwise')(x)

    x = layers.BatchNormalization( axis=-1, epsilon=1e-3, momentum=0.999, \
        name= str(id) + '_depthwise_bn')(x)

    x = layers.ReLU(name= str(id) + '_relu')(x)

    pred = layers.Conv2D(filters, kernel_size=1, strides=1, use_bias=False, \
        padding='same', name= str(id) + '_pred_cnv')(x)

    x = layers.BatchNormalization( axis=-1, epsilon=1e-3, momentum=0.999, \
        name= str(id) + '_pred_bn')(pred)

    x = layers.ReLU(name= str(id) + '_pred_relu')(x)

    return x, pred


def generate_grid(x,layer_scale,ratios,name):
    """
        creates a grid for a feature layer and then reshapes them \
        into a grid of boxes
    """  
    # creating a grid for 
    num_boxes = len(ratios)

    x_grid = (tf.range(0,x.shape[1],dtype=tf.float32)+0.5)/x.shape[1]
    y_grid = (tf.range(0,x.shape[2],dtype=tf.float32)+0.5)/x.shape[2]

    x_grid = tf.expand_dims(x_grid,axis=0)
    y_grid = tf.expand_dims(y_grid,axis=1)

    x_grid = tf.tile(x_grid,[x.shape[2],1])
    y_grid = tf.tile(y_grid,[1,x.shape[1]])

    empty_box = tf.zeros([1,x.shape[1],x.shape[2]])

    # 1 * M * N
    x_grid = tf.expand_dims(x_grid,axis=2)
    y_grid = tf.expand_dims(y_grid,axis=2)

    # Account for background class
    empty_box = tf.tile(empty_box, [NUM_CLASSES+1,1,1])
    empty_box = tf.transpose(empty_box,[1,2,0])
    
    scale_ = tf.tile([layer_scale],[x.shape[1]])
    scale_ = tf.expand_dims(scale_,axis=0)
    scale_grid = tf.tile(scale_,[x.shape[2],1])
    scale_grid = tf.expand_dims(scale_grid,axis=2)

    # Account for different scales
    width = scale_grid*tf.sqrt(ratios)
    height= scale_grid/tf.sqrt(ratios)

    #  Num AXES               11      , 1   , 1      ,num_boxes*2
    box_const = tf.concat([empty_box,x_grid,y_grid,width,height],axis=2)

    assert box_const.shape[2]==NUM_CLASSES + 1 + 2 + num_boxes*2


    # Repeats the first 10 classes as zeros, since the class idxs must not be changed
    # For each box, this code copies the first 11 values in box_const which are zeros
    # Then it picks one x coordinate grid, one y coordinate grid and according to 
    # the corrsponding ratio, one width and height
    # gathering idxs become 
    # [0,1,2...,10,11,12,13,13+num_boxes, 0,1,2,....,10,11,12,13+1,13+num_boxes+1,0,1,2... ]
    box_idx = tf.concat([tf.concat([tf.range(0,NUM_CLASSES+3,1), \
        [NUM_CLASSES+3+i], [NUM_CLASSES+3+num_boxes+i] ],axis=0) \
            for i in range(num_boxes)],axis=0)
    
    # transpose for easy gathering
    box_const = tf.transpose(box_const, [2,0,1])
    const_add = tf.gather(box_const,box_idx)
    # transpose back to m * n * 1 format
    const_add = tf.transpose(const_add, [1,2,0])

    const_add = tf.expand_dims(const_add,axis=0)

    const_add = tf.compat.v1.placeholder_with_default(const_add, \
        [None,x.shape[1],x.shape[2],num_boxes*NUM_PREDS_PER_BOX])

    box_reshape = layers.Reshape((const_add.shape[1]*const_add.shape[2],const_add.shape[3]),\
        name=name+"_get_k_grids")(const_add)

    box_reshape = layers.Reshape((box_reshape.shape[1]*box_reshape.shape[2]// \
        (NUM_PREDS_PER_BOX), NUM_PREDS_PER_BOX), \
            name=name+"_remove_k_grid_axis")(box_reshape)

    return box_reshape


def box_prediction_layer(x,layer_scale,ratios,name):
    """
    Takes a particular layers features, convolves and obtains boxes for
    an entire layer, then reshapes them into boxes.
    """  
    num_boxes = len(ratios)
    y = layers.Conv2D(num_boxes*(NUM_PREDS_PER_BOX), \
        padding='same',kernel_size=3, name=name+"_box_conv")(x)

    box_reshape = layers.Reshape((y.shape[1]*y.shape[2],y.shape[3]),\
        name=name+"_get_k_grids")(y)

    bbox_pred = layers.Reshape((box_reshape.shape[1]*box_reshape.shape[2]// \
        (NUM_PREDS_PER_BOX), NUM_PREDS_PER_BOX), \
            name=name+"_remove_k_grid_axis")(box_reshape)

    return bbox_pred


def predict_bboxes(features, layer_names, layer_scales, layer_ratios):
    '''
    After extracting features from the model head, calculates boxes for
    each
    '''
    preds = []

    for x,y,z in zip(features,layer_scales,layer_names):
        pred  = box_prediction_layer(x=x,layer_scale=y, \
            ratios=layer_ratios[z],name=z)
        
        preds.append(pred)

    return tf.concat(preds,axis=1) #, tf.concat(grids,axis=1)


def get_grid(features, layer_names, layer_scales, layer_ratios):
    '''
    Calculates the default grid offsets
    '''
    grids = []
    for x,y,z in zip(features,layer_scales,layer_names):
        grid  = generate_grid(x=x,layer_scale=y, \
            ratios=layer_ratios[z],name=z)

        grids.append(grid)

    return  tf.concat(grids,axis=1)


def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def correct_pad(inputs, kernel_size):
  """Returns a tuple for zero-padding for 2D convolution with downsampling.
  Arguments:
    inputs: Input tensor.
    kernel_size: An integer or tuple/list of 2 integers.
  Returns:
    A tuple.
  """
  img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
  input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]
  if isinstance(kernel_size, int):
    kernel_size = (kernel_size, kernel_size)
  if input_size[0] is None:
    adjust = (1, 1)
  else:
    adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
  correct = (kernel_size[0] // 2, kernel_size[1] // 2)
  return ((correct[0] - adjust[0], correct[0]),
          (correct[1] - adjust[1], correct[1]))