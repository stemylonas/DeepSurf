"""Cuda op Python library."""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

#import os.path

#import numpy as np

import tensorflow as tf

    
_amul_module = tf.load_op_library('/home/smylonas/Desktop/ellidek/code/my_lds/lib/amul_kernel_cube.so')
amul = _amul_module.amul

_amul_module_backward = tf.load_op_library('/home/smylonas/Desktop/ellidek/code/my_lds/lib/amul_kernel_cube_grad.so')
amul_backward = _amul_module_backward.amul_backward


from tensorflow.python.framework import ops
#from tensorflow.python.ops import array_ops
#from tensorflow.python.ops import sparse_ops

@ops.RegisterGradient("Amul")
def _amul_grad(op, grad):

  inp = op.inputs[0]
  weight = op.inputs[1]
  
  #axis = op.get_attr('axis')

  [input_grad, weight_grad] = amul_backward(inp, weight, grad) 
  #[input_grad, weight_grad] = amul_backward(inp, weight, grad, axis=axis) 
  return [input_grad, weight_grad]  # List of two Tensor, since we have two inputs 
  