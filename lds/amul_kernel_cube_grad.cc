/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("AmulBackward")
	.Input("input: float")
	.Input("weight: float")
    .Input("grad_output: float")
    .Output("grad_input: float")
	.Output("grad_weight: float")
	.SetShapeFn([](shape_inference::InferenceContext* c) {  
	  c->set_output(0, c->input(0));
	  c->set_output(1, c->input(1));
      return Status::OK();
    })
	;

void AmulBackwardKernelLauncher(const float* input, const float* weight, const float* gradOutput, float* gradInput, float* gradWeight,
                                const int batchSize, const int input_xSize, const int input_ySize, const int input_zSize, const int numChannels);

class AmulBackwardOp : public OpKernel {
 public:
  explicit AmulBackwardOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
   
	// Grab the input tensor
	
	const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();
	
	const Tensor& weight_tensor = context->input(1);
    auto weight = weight_tensor.flat<float>();
	
	const Tensor& grad_output_tensor = context->input(2);
    auto grad_output = grad_output_tensor.flat<float>();
	

	// calculate output tensor shape
	
	int batchSize = input_tensor.shape().dim_size(0);
	int input_xSize = input_tensor.shape().dim_size(1);
	int input_ySize = input_tensor.shape().dim_size(2);
    int input_zSize = input_tensor.shape().dim_size(3);
	int numChannels = input_tensor.shape().dim_size(4);
		
	// Create an output tensor
	Tensor* gradInput_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &gradInput_tensor));
    auto grad_input = gradInput_tensor->template flat<float>();
	
	//TensorShape gradWeight_shape;
	//gradWeight_shape.AddDim(numChannels);
	//gradWeight_shape.AddDim(3);
	//gradWeight_shape.AddDim(3);	
	Tensor* gradWeight_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, weight_tensor.shape(), &gradWeight_tensor));
    auto grad_weight = gradWeight_tensor->template flat<float>();
		
    // Call the cuda kernel launcher
    AmulBackwardKernelLauncher(input.data(), weight.data(), grad_output.data(), grad_input.data(), grad_weight.data(), batchSize, input_xSize, input_ySize, input_zSize, numChannels);
	
  }
};

REGISTER_KERNEL_BUILDER(Name("AmulBackward").Device(DEVICE_GPU), AmulBackwardOp);
