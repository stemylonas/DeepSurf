"""
Created on Thu Nov  1 12:04:04 2018

@author: smylonas
"""

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from net import resnet_3d_utils

resnet_arg_scope = resnet_3d_utils.resnet_arg_scope


@add_arg_scope
def resid_unit(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               outputs_collections=None,
               scope=None):
  """Residual unit with BN after convolutions.
  This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
  its definition. 
  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.
  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.
  Returns:
    The ResNet unit's output.
  """
  with variable_scope.variable_scope(scope, 'resid_v1', [inputs]) as sc:
   # print (inputs.shape)
    depth_in = utils.last_dimension(inputs.get_shape(), min_rank=5)
    if depth == depth_in:
      shortcut = resnet_3d_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = layers.conv3d(
          inputs,
          depth, [1, 1, 1],
          stride=stride,
          activation_fn=None,
          scope='shortcut')

    residual = resnet_3d_utils.conv3d_same(inputs, depth_bottleneck, 3, stride=1, scope='conv1')   
    residual = layers.conv3d(residual, depth_bottleneck, 3, stride, scope='conv2')

    output = nn_ops.relu(shortcut + residual)

    return utils.collect_named_outputs(outputs_collections, sc.name, output)


def resnet_v1(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              reuse=None,
              scope=None):
  """Generator for v1 ResNet models.
  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a resnet_utils.Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks. If None
      we return the features before the logit layer.
    is_training: whether batch_norm layers are in training mode.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    include_root_block: If True, include the initial convolution followed by
      max-pooling, if False excludes it.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.
  Raises:
    ValueError: If the target output_stride is not valid.
  """
  with variable_scope.variable_scope(
      scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with arg_scope(
        [layers.conv3d, resid_unit, resnet_3d_utils.stack_blocks_dense],
        outputs_collections=end_points_collection):
      with arg_scope([layers.batch_norm], is_training=is_training):
        net = inputs
        net = resnet_3d_utils.stack_blocks_dense(net, blocks, output_stride)
        if global_pool:
          net = math_ops.reduce_mean(net, [1, 2, 3], name='pool5', keepdims=True)
        if num_classes is not None:
          net = layers.conv3d(
              net,
              num_classes, [1, 1, 1],
              activation_fn=None,
              normalizer_fn=None,
              scope='logits')
          
        # Convert end_points_collection into a dictionary of end_points.
        end_points = utils.convert_collection_to_dict(end_points_collection)
        if num_classes is not None and num_classes != 1:
          end_points['predictions'] = layers_lib.softmax(net, scope='predictions')
          net = tf.squeeze(net)
        elif num_classes == 1:
          net = tf.squeeze(net)                                                    
          end_points['probs'] = tf.nn.sigmoid(net)
       
        return net, end_points


def resnet_v1_block(scope, depth_out, num_units, stride):
  """Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.
  Returns:
    A resnet_v1 bottleneck block.
  """
  return resnet_3d_utils.Block(scope, resid_unit, [{
      'depth': depth_out,
      'depth_bottleneck': depth_out,
      'stride': 1
  }] * (num_units - 1) + [{
      'depth': depth_out,
      'depth_bottleneck': depth_out,
      'stride': stride
  }])


def resnet_v1_18(inputs,
                 num_classes,
                 is_training,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 scope='resnet3d_v1_18'):
  blocks = [
      resnet_v1_block('block1', depth_out=64, num_units=2, stride=2),
      resnet_v1_block('block2', depth_out=128, num_units=2, stride=2),
      resnet_v1_block('block3', depth_out=256, num_units=2, stride=2),
      resnet_v1_block('block4', depth_out=512, num_units=2, stride=1) 
  ]
  return resnet_v1(
      inputs,
      blocks,
      num_classes,
      is_training,
      global_pool,
      output_stride,
      include_root_block=False,
      reuse=reuse,
      scope=scope)

