
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"


using namespace tensorflow;
using namespace tensorflow::shape_inference;

REGISTER_OP("Amul")
    .Input("input: float")
	.Input("weight: float")
    .Output("output: float")
	.SetShapeFn([](InferenceContext* c) {
   
      std::vector<DimensionHandle> dim(5);
	  DimensionHandle dimtemp;
              
      dim[0] = c->Dim(c->input(0),0);
	  dim[4] = c->Dim(c->input(0),4);
	  
      c->Subtract(c->Dim(c->input(0),1), 2, &dimtemp);
	  c->Multiply(dimtemp, 3, &dim[1]);
     
	  c->Subtract(c->Dim(c->input(0),2), 2, &dimtemp);
	  c->Multiply(dimtemp, 3, &dim[2]);
	  
	  c->Subtract(c->Dim(c->input(0),3), 2, &dimtemp);
	  c->Multiply(dimtemp, 3, &dim[3]);

	  ShapeHandle outputShape = c->MakeShape(dim);
	  
	  c->set_output(0, outputShape);
      return tensorflow::Status::OK();
    })
;


void AmulKernelLauncher(const float* input, const float* weight, float* output, const int batchSize, 
    const int input_xSize, const int input_ySize, const int input_zSize, const int numChannels);

class AmulOp : public OpKernel {
 public:
  explicit AmulOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    
	// Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input_flat = input_tensor.flat<float>();
	
	const Tensor& weight_tensor = context->input(1);
    auto weight = weight_tensor.flat<float>();	
	
	
	int batchSize = input_tensor.shape().dim_size(0);
	int input_xSize = input_tensor.shape().dim_size(1);
	int input_ySize = input_tensor.shape().dim_size(2);
    int input_zSize = input_tensor.shape().dim_size(3);
	int numChannels = input_tensor.shape().dim_size(4);
	
	//int N = batchSize * input_xSize * input_ySize * numChannels;
    
	//printf("\n\n%d %d %d %d \n\n", batchSize,input_xSize, input_ySize, numChannels);

	TensorShape output_tensor_shape;
         
	output_tensor_shape.AddDim(batchSize);
    output_tensor_shape.AddDim(3 * (input_xSize - 2));
	output_tensor_shape.AddDim(3 * (input_ySize - 2));
    output_tensor_shape.AddDim(3 * (input_zSize - 2));
                    
	output_tensor_shape.AddDim(numChannels);
		
	// Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_tensor_shape, &output_tensor));
    auto output_flat = output_tensor->template flat<float>();

    // Call the cuda kernel launcher
    AmulKernelLauncher(input_flat.data(), weight.data(), output_flat.data(), batchSize, input_xSize, input_ySize, input_zSize, numChannels);
	
  }
};

REGISTER_KERNEL_BUILDER(Name("Amul").Device(DEVICE_GPU), AmulOp);
