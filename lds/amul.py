"""Cuda op Python library."""

import tensorflow as tf

    
_amul_module = tf.load_op_library('lds/lib/amul_kernel_cube.so')
amul = _amul_module.amul

_amul_module_backward = tf.load_op_library('lds/lib/amul_kernel_cube_grad.so')
amul_backward = _amul_module_backward.amul_backward


from tensorflow.python.framework import ops

@ops.RegisterGradient("Amul")
def _amul_grad(op, grad):

  inp = op.inputs[0]
  weight = op.inputs[1]

  [input_grad, weight_grad] = amul_backward(inp, weight, grad) 
 
  return [input_grad, weight_grad]  # List of two Tensor, since we have two inputs 
  
