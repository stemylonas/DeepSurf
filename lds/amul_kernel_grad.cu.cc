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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

__global__ void AmulBackwardInputKernel_x(const float* weight, const float* gradOutput, float* gradInput, const int batchSize, 
                                        const int gradInput_xSize, const int gradInput_ySize, const int gradInput_zSize, const int numChannels) { 

    //int blockDim = 10;                                    
                    
	int b = blockIdx.x/numChannels;
	int ch = blockIdx.x - b * numChannels;
	
	int x = threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
    
	float a00,a01,a02,a10,a11,a12,a20,a21,a22; 
	float gradX0, gradX1, gradX2;
	float gradXest0, gradXest1, gradXest2;
	
	int gradInputBatchStride = b * gradInput_xSize * gradInput_ySize * gradInput_zSize * numChannels;
	
	int gradOutput_xSize = gradInput_xSize;
    int gradOutput_ySize = (gradInput_ySize-2) * 3;
    int gradOutput_zSize = (gradInput_zSize-2) * 3;
    
	int gradOutputBatchStride = b * gradOutput_xSize * gradOutput_ySize * gradOutput_zSize * numChannels;
	
	// retrieve a
	a00 = *weight;
	a01 = *(weight+1);
	a02 = *(weight+2);
	a10 = *(weight+3);
	a11 = *(weight+4);
	a12 = *(weight+5);
	a20 = *(weight+6);
	a21 = *(weight+7);
	a22 = *(weight+8);
		
		
	if (x < gradInput_xSize && y < (gradInput_ySize-2) && z < (gradInput_zSize-2)) {
		
		float * gradInputAddr = gradInput + gradInputBatchStride + (x*gradInput_ySize*gradInput_zSize + y*gradInput_zSize + z)*numChannels + ch;
		
		for (int k=0; k<3; k++){			

        		int gradXestStride = gradOutputBatchStride + (x*gradOutput_ySize*gradOutput_zSize + 3*y*gradOutput_zSize + (3*z+k))*numChannels + ch;
							
				gradXest0 = gradOutput[gradXestStride];
				gradXest1 = gradOutput[gradXestStride+gradOutput_zSize*numChannels];
				gradXest2 = gradOutput[gradXestStride+2*gradOutput_zSize*numChannels];
				
				gradX0 = a00*gradXest0 + a10*gradXest1 + a20*gradXest2;
				gradX1 = a01*gradXest0 + a11*gradXest1 + a21*gradXest2;
				gradX2 = a02*gradXest0 + a12*gradXest1 + a22*gradXest2;
				
				float * gradInputAddr0 = gradInputAddr + k*numChannels;
				
				atomicAdd(gradInputAddr0, gradX0);
				atomicAdd(gradInputAddr0 + gradInput_zSize*numChannels, gradX1);
				atomicAdd(gradInputAddr0 + 2*gradInput_zSize*numChannels, gradX2);			
		}
	}


}

__global__ void AmulBackwardInputKernel_y(const float* weight, const float* gradOutput, float* gradInput, const int batchSize, 
                                        const int gradInput_xSize, const int gradInput_ySize, const int gradInput_zSize, const int numChannels) { 

    //int blockDim = 10;                                    
                    
	int b = blockIdx.x/numChannels;
	int ch = blockIdx.x - b * numChannels;
	
	int x = blockIdx.y * blockDim.x + threadIdx.x;
    int y = threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
    
	float a00,a01,a02,a10,a11,a12,a20,a21,a22; 
	float gradX0, gradX1, gradX2;
	float gradXest0, gradXest1, gradXest2;
	
	int gradInputBatchStride = b * gradInput_xSize * gradInput_ySize * gradInput_zSize * numChannels;
	
	int gradOutput_xSize = (gradInput_xSize-2) * 3;
    int gradOutput_ySize = gradInput_ySize;
    int gradOutput_zSize = (gradInput_zSize-2) * 3;
    
	int gradOutputBatchStride = b * gradOutput_xSize * gradOutput_ySize * gradOutput_zSize * numChannels;
	
	// retrieve a
	a00 = *weight;
	a01 = *(weight+1);
	a02 = *(weight+2);
	a10 = *(weight+3);
	a11 = *(weight+4);
	a12 = *(weight+5);
	a20 = *(weight+6);
	a21 = *(weight+7);
	a22 = *(weight+8);
		
		
	if (x < (gradInput_xSize-2) && y < gradInput_ySize && z < (gradInput_zSize-2)) {
		
		float * gradInputAddr = gradInput + gradInputBatchStride + (x*gradInput_ySize*gradInput_zSize + y*gradInput_zSize + z)*numChannels + ch;
		
		for (int k=0; k<3; k++){			

        		int gradXestStride = gradOutputBatchStride + (3*x*gradOutput_ySize*gradOutput_zSize + y*gradOutput_zSize + (3*z+k))*numChannels + ch;
							
				gradXest0 = gradOutput[gradXestStride];
				gradXest1 = gradOutput[gradXestStride+gradOutput_ySize*gradOutput_zSize*numChannels];
				gradXest2 = gradOutput[gradXestStride+2*gradOutput_ySize*gradOutput_zSize*numChannels];
				
				gradX0 = a00*gradXest0 + a10*gradXest1 + a20*gradXest2;
				gradX1 = a01*gradXest0 + a11*gradXest1 + a21*gradXest2;
				gradX2 = a02*gradXest0 + a12*gradXest1 + a22*gradXest2;
				
				float * gradInputAddr0 = gradInputAddr + k*numChannels;
				
				atomicAdd(gradInputAddr0, gradX0);
				atomicAdd(gradInputAddr0 + gradInput_ySize*gradInput_zSize*numChannels, gradX1);
				atomicAdd(gradInputAddr0 + 2*gradInput_ySize*gradInput_zSize*numChannels, gradX2);			
		}
	}


}

__global__ void AmulBackwardInputKernel_z(const float* weight, const float* gradOutput, float* gradInput, const int batchSize, 
                                        const int gradInput_xSize, const int gradInput_ySize, const int gradInput_zSize, const int numChannels) { 

    //int blockDim = 10;                                    
                    
	int b = blockIdx.x/numChannels;
	int ch = blockIdx.x - b * numChannels;
	
	int x = blockIdx.y * blockDim.x + threadIdx.x;
	int y = blockIdx.z * blockDim.y + threadIdx.y;
    int z  = threadIdx.z;
	
	float a00,a01,a02,a10,a11,a12,a20,a21,a22; 
	float gradX0, gradX1, gradX2;
	float gradXest0, gradXest1, gradXest2;
	
	int gradInputBatchStride = b * gradInput_xSize * gradInput_ySize * gradInput_zSize * numChannels;
	
	int gradOutput_xSize = (gradInput_xSize-2) * 3;
    int gradOutput_ySize = (gradInput_ySize-2) * 3;
    int gradOutput_zSize = gradInput_zSize;
                           
	int gradOutputBatchStride = b * gradOutput_xSize * gradOutput_ySize * gradOutput_zSize * numChannels;
	
	// retrieve a
	a00 = *weight;
	a01 = *(weight+1);
	a02 = *(weight+2);
	a10 = *(weight+3);
	a11 = *(weight+4);
	a12 = *(weight+5);
	a20 = *(weight+6);
	a21 = *(weight+7);
	a22 = *(weight+8);
		
		
	if (x < (gradInput_xSize-2) && y < (gradInput_ySize-2) && z < gradInput_zSize) {
		
		float * gradInputAddr = gradInput + gradInputBatchStride + (x*gradInput_ySize*gradInput_zSize + y*gradInput_zSize + z)*numChannels + ch;
		
		for (int j=0; j<3; j++){			

        		int gradXestStride = gradOutputBatchStride + (3*x*gradOutput_ySize*gradOutput_zSize + (3*y+j)*gradOutput_zSize + z)*numChannels + ch;
							
				gradXest0 = gradOutput[gradXestStride];
				gradXest1 = gradOutput[gradXestStride+gradOutput_ySize*gradOutput_zSize*numChannels];
				gradXest2 = gradOutput[gradXestStride+2*gradOutput_ySize*gradOutput_zSize*numChannels];
				
				gradX0 = a00*gradXest0 + a10*gradXest1 + a20*gradXest2; // why transpose and not inverse??
				gradX1 = a01*gradXest0 + a11*gradXest1 + a21*gradXest2;
				gradX2 = a02*gradXest0 + a12*gradXest1 + a22*gradXest2;
				
				float * gradInputAddr0 = gradInputAddr + j*gradInput_zSize*numChannels;
				
				atomicAdd(gradInputAddr0, gradX0);
				atomicAdd(gradInputAddr0 + gradInput_ySize*gradInput_zSize*numChannels, gradX1);
				atomicAdd(gradInputAddr0 + 2*gradInput_ySize*gradInput_zSize*numChannels, gradX2);			
		}
	}


}

__global__ void ZeroGradInputKernel(float *gradInput, int N){
	
	for (int i=threadIdx.x; i<N; i+=blockDim.x){
		gradInput[i] = 0.0f;
	}
	
}

__global__ void ZeroGradWeightKernel(float *gradWeight, int N){
	
	for (int i=0; i<N; i++){
		gradWeight[i] = 0.0f;
	}
	
}

__global__ void AmulBackwardWeightKernel_x(const float *input, const float *gradOutput, float *grad_a, const int batchSize, 
                                         const int input_xSize, const int input_ySize, const int input_zSize, const int numChannels){
	
    //int blockDim = 10;                                     
                                         
	int b = blockIdx.x/numChannels;
	int ch = blockIdx.x - b * numChannels;
	
	int x = threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
    
	
	extern __shared__ float mem[];
	
	float grad_a00,grad_a01,grad_a02,grad_a10,grad_a11,grad_a12,grad_a20,grad_a21,grad_a22;
	
	float gradXest00,gradXest01,gradXest02,gradXest10,gradXest11,gradXest12,gradXest20,gradXest21,gradXest22; 
	
	int inputBatchStride = b * input_xSize * input_ySize * input_zSize * numChannels;	
	
	int gradOutput_xSize = input_xSize;
    int gradOutput_ySize = (input_ySize-2) * 3;
	int gradOutput_zSize = (input_zSize-2) * 3;
    
	int gradOutputBatchStride = b * gradOutput_xSize * gradOutput_ySize * gradOutput_zSize * numChannels;
	
	for (int j=0; j<blockDim.y+2; j+=blockDim.y)
		for (int k=0; k<blockDim.z+2; k+=blockDim.z)
			if (j+threadIdx.y<blockDim.y+2 && k+threadIdx.z<blockDim.z+2 && (y+j)<input_ySize && (z+k)<input_zSize)
				mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + (threadIdx.y+j)*(blockDim.z+2) + threadIdx.z+k] =
					input[inputBatchStride + (x*input_ySize*input_zSize + (y+j)*input_zSize+z+k)*numChannels + ch];
	__syncthreads();
	
	if (x < input_xSize && y < (input_ySize-2) && z < (input_zSize-2) ) {
	
		int gradXestStride0 = gradOutputBatchStride + (x*gradOutput_ySize*gradOutput_zSize + 3*y*gradOutput_zSize + 3*z)*numChannels + ch;
		
		gradXest00 = gradOutput[gradXestStride0];
		gradXest01 = gradOutput[gradXestStride0+1*numChannels];
		gradXest02 = gradOutput[gradXestStride0+2*numChannels];
		gradXest10 = gradOutput[gradXestStride0+gradOutput_zSize*numChannels];
		gradXest11 = gradOutput[gradXestStride0+(gradOutput_zSize+1)*numChannels];
		gradXest12 = gradOutput[gradXestStride0+(gradOutput_zSize+2)*numChannels];
		gradXest20 = gradOutput[gradXestStride0+(2*gradOutput_zSize)*numChannels];
		gradXest21 = gradOutput[gradXestStride0+(2*gradOutput_zSize+1)*numChannels];
		gradXest22 = gradOutput[gradXestStride0+(2*gradOutput_zSize+2)*numChannels];
		
		
		grad_a00 = gradXest00 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z] +
				   gradXest01 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+1] +
				   gradXest02 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+2];
		grad_a01 = gradXest00 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + (threadIdx.y+1)*(blockDim.z+2) + threadIdx.z] +
				   gradXest01 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + (threadIdx.y+1)*(blockDim.z+2) + threadIdx.z+1] +
				   gradXest02 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + (threadIdx.y+1)*(blockDim.z+2) + threadIdx.z+2];
		grad_a02 = gradXest00 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + (threadIdx.y+2)*(blockDim.z+2) + threadIdx.z] +
				   gradXest01 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + (threadIdx.y+2)*(blockDim.z+2) + threadIdx.z+1] +
				   gradXest02 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + (threadIdx.y+2)*(blockDim.z+2) + threadIdx.z+2];
		
		grad_a10 = gradXest10 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z] +
				   gradXest11 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+1] +
				   gradXest12 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+2];
		grad_a11 = gradXest10 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + (threadIdx.y+1)*(blockDim.z+2) + threadIdx.z] +
				   gradXest11 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + (threadIdx.y+1)*(blockDim.z+2) + threadIdx.z+1] +
				   gradXest12 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + (threadIdx.y+1)*(blockDim.z+2) + threadIdx.z+2];
		grad_a12 = gradXest10 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + (threadIdx.y+2)*(blockDim.z+2) + threadIdx.z] +
				   gradXest11 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + (threadIdx.y+2)*(blockDim.z+2) + threadIdx.z+1] +
				   gradXest12 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + (threadIdx.y+2)*(blockDim.z+2) + threadIdx.z+2];
		
		grad_a20 = gradXest20 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z] +
				   gradXest21 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+1] +
				   gradXest22 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+2];
		grad_a21 = gradXest20 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + (threadIdx.y+1)*(blockDim.z+2) + threadIdx.z] +
				   gradXest21 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + (threadIdx.y+1)*(blockDim.z+2) + threadIdx.z+1] +
				   gradXest22 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + (threadIdx.y+1)*(blockDim.z+2) + threadIdx.z+2];
		grad_a22 = gradXest20 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + (threadIdx.y+2)*(blockDim.z+2) + threadIdx.z] +
				   gradXest21 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + (threadIdx.y+2)*(blockDim.z+2) + threadIdx.z+1] +
				   gradXest22 * mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + (threadIdx.y+2)*(blockDim.z+2) + threadIdx.z+2];
	
		// accumulate gradients
		atomicAdd(grad_a, grad_a00);
		atomicAdd(grad_a+1, grad_a01);
		atomicAdd(grad_a+2, grad_a02);
		atomicAdd(grad_a+3, grad_a10);
		atomicAdd(grad_a+4, grad_a11);
		atomicAdd(grad_a+5, grad_a12);
		atomicAdd(grad_a+6, grad_a20);
		atomicAdd(grad_a+7, grad_a21);
		atomicAdd(grad_a+8, grad_a22);
		
	}
}

__global__ void AmulBackwardWeightKernel_y(const float *input, const float *gradOutput, float *grad_a, const int batchSize, 
                                         const int input_xSize, const int input_ySize, const int input_zSize, const int numChannels){
	
    //int blockDim = 10;                                     
                                         
	int b = blockIdx.x/numChannels;
	int ch = blockIdx.x - b * numChannels;
	
	int x = blockIdx.y * blockDim.x + threadIdx.x;
    int y = threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
    
	
	extern __shared__ float mem[];
	
	float grad_a00,grad_a01,grad_a02,grad_a10,grad_a11,grad_a12,grad_a20,grad_a21,grad_a22;
	
	float gradXest00,gradXest01,gradXest02,gradXest10,gradXest11,gradXest12,gradXest20,gradXest21,gradXest22; 
	
	int inputBatchStride = b * input_xSize * input_ySize * input_zSize * numChannels;	
	
	int gradOutput_xSize = (input_xSize-2) * 3;
    int gradOutput_ySize = input_ySize;
	int gradOutput_zSize = (input_zSize-2) * 3;
    
	int gradOutputBatchStride = b * gradOutput_xSize * gradOutput_ySize * gradOutput_zSize * numChannels;
	
	for (int i=0; i<blockDim.x+2; i+=blockDim.x)
		for (int k=0; k<blockDim.z+2; k+=blockDim.z)
			if (i+threadIdx.x<blockDim.x+2 && k+threadIdx.z<blockDim.z+2 && (x+i)<input_xSize && (z+k)<input_zSize)
				mem[(i+threadIdx.x)*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+k] =
					input[inputBatchStride + ((x+i)*input_ySize*input_zSize + y*input_zSize+z+k)*numChannels + ch];
	__syncthreads();
	 
	if (x < (input_xSize-2) && y < input_ySize && z < (input_zSize-2) ) {
	
		int gradXestStride0 = gradOutputBatchStride + (3*x*gradOutput_ySize*gradOutput_zSize + y*gradOutput_zSize + 3*z)*numChannels + ch;
		
		gradXest00 = gradOutput[gradXestStride0];
		gradXest01 = gradOutput[gradXestStride0+1*numChannels];
		gradXest02 = gradOutput[gradXestStride0+2*numChannels];
		gradXest10 = gradOutput[gradXestStride0+gradOutput_ySize*gradOutput_zSize*numChannels];
		gradXest11 = gradOutput[gradXestStride0+(gradOutput_ySize*gradOutput_zSize+1)*numChannels];
		gradXest12 = gradOutput[gradXestStride0+(gradOutput_ySize*gradOutput_zSize+2)*numChannels];
		gradXest20 = gradOutput[gradXestStride0+(2*gradOutput_ySize*gradOutput_zSize)*numChannels];
		gradXest21 = gradOutput[gradXestStride0+(2*gradOutput_ySize*gradOutput_zSize+1)*numChannels];
		gradXest22 = gradOutput[gradXestStride0+(2*gradOutput_ySize*gradOutput_zSize+2)*numChannels];
		
		
		grad_a00 = gradXest00 * mem[threadIdx.x*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z] +
				   gradXest01 * mem[threadIdx.x*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+1] +
				   gradXest02 * mem[threadIdx.x*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+2];
		grad_a01 = gradXest00 * mem[(threadIdx.x+1)*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z] +
				   gradXest01 * mem[(threadIdx.x+1)*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+1] +
				   gradXest02 * mem[(threadIdx.x+1)*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+2];
		grad_a02 = gradXest00 * mem[(threadIdx.x+2)*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z] +
				   gradXest01 * mem[(threadIdx.x+2)*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+1] +
				   gradXest02 * mem[(threadIdx.x+2)*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+2];
		
		grad_a10 = gradXest10 * mem[threadIdx.x*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z] +
				   gradXest11 * mem[threadIdx.x*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+1] +
				   gradXest12 * mem[threadIdx.x*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+2];
		grad_a11 = gradXest10 * mem[(threadIdx.x+1)*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z] +
				   gradXest11 * mem[(threadIdx.x+1)*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+1] +
				   gradXest12 * mem[(threadIdx.x+1)*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+2];
		grad_a12 = gradXest10 * mem[(threadIdx.x+2)*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z] +
				   gradXest11 * mem[(threadIdx.x+2)*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+1] +
				   gradXest12 * mem[(threadIdx.x+2)*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+2];
		
		grad_a20 = gradXest20 * mem[threadIdx.x*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z] +
				   gradXest21 * mem[threadIdx.x*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+1] +
				   gradXest22 * mem[threadIdx.x*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+2];
		grad_a21 = gradXest20 * mem[(threadIdx.x+1)*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z] +
				   gradXest21 * mem[(threadIdx.x+1)*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+1] +
				   gradXest22 * mem[(threadIdx.x+1)*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+2];
		grad_a22 = gradXest20 * mem[(threadIdx.x+2)*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z] +
				   gradXest21 * mem[(threadIdx.x+2)*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+1] +
				   gradXest22 * mem[(threadIdx.x+2)*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + threadIdx.z+2];
	
		// accumulate gradients
		atomicAdd(grad_a, grad_a00);
		atomicAdd(grad_a+1, grad_a01);
		atomicAdd(grad_a+2, grad_a02);
		atomicAdd(grad_a+3, grad_a10);
		atomicAdd(grad_a+4, grad_a11);
		atomicAdd(grad_a+5, grad_a12);
		atomicAdd(grad_a+6, grad_a20);
		atomicAdd(grad_a+7, grad_a21);
		atomicAdd(grad_a+8, grad_a22);
		
	}
}

__global__ void AmulBackwardWeightKernel_z(const float *input, const float *gradOutput, float *grad_a, const int batchSize, 
                                         const int input_xSize, const int input_ySize, const int input_zSize, const int numChannels){
	
    //int blockDim = 10;                                     
                                         
	int b = blockIdx.x/numChannels;
	int ch = blockIdx.x - b * numChannels;
	
	int x = blockIdx.y * blockDim.x + threadIdx.x;
	int y = blockIdx.z * blockDim.y + threadIdx.y;
    int z  = threadIdx.z;
	
	extern __shared__ float mem[];
	
	float grad_a00,grad_a01,grad_a02,grad_a10,grad_a11,grad_a12,grad_a20,grad_a21,grad_a22;
	
	float gradXest00,gradXest01,gradXest02,gradXest10,gradXest11,gradXest12,gradXest20,gradXest21,gradXest22; 
	
	int inputBatchStride = b * input_xSize * input_ySize * input_zSize * numChannels;	
	
	int gradOutput_xSize = (input_xSize-2) * 3;
	int gradOutput_ySize = (input_ySize-2) * 3;
    int gradOutput_zSize = input_zSize;
	int gradOutputBatchStride = b * gradOutput_xSize * gradOutput_ySize * gradOutput_zSize * numChannels;
	
	for (int i=0; i<blockDim.x+2; i+=blockDim.x)
		for (int j=0; j<blockDim.y+2; j+=blockDim.y)
			if (i+threadIdx.x<blockDim.x+2 && j+threadIdx.y<blockDim.y+2 && (x+i)<input_xSize && (y+j)<input_ySize)
				mem[(i+threadIdx.x)*(blockDim.y+2)*input_zSize + (j+threadIdx.y)*input_zSize + threadIdx.z] =
					input[inputBatchStride + ((x+i)*input_ySize*input_zSize + (y+j)*input_zSize+z)*numChannels + ch];
	__syncthreads();
	
	if (x < (input_xSize-2) && y < (input_ySize-2) && z < input_zSize ) {
	
		int gradXestStride0 = gradOutputBatchStride + (3*x*gradOutput_ySize*gradOutput_zSize + 3*y*gradOutput_zSize + z)*numChannels + ch;
		
		gradXest00 = gradOutput[gradXestStride0];
		gradXest01 = gradOutput[gradXestStride0+1*gradOutput_zSize*numChannels];
		gradXest02 = gradOutput[gradXestStride0+2*gradOutput_zSize*numChannels];
		gradXest10 = gradOutput[gradXestStride0+gradOutput_ySize*gradOutput_zSize*numChannels];
		gradXest11 = gradOutput[gradXestStride0+(gradOutput_ySize+1)*gradOutput_zSize*numChannels];
		gradXest12 = gradOutput[gradXestStride0+(gradOutput_ySize+2)*gradOutput_zSize*numChannels];
		gradXest20 = gradOutput[gradXestStride0+(2*gradOutput_ySize)*gradOutput_zSize*numChannels];
		gradXest21 = gradOutput[gradXestStride0+(2*gradOutput_ySize+1)*gradOutput_zSize*numChannels];
		gradXest22 = gradOutput[gradXestStride0+(2*gradOutput_ySize+2)*gradOutput_zSize*numChannels];
		
		
		grad_a00 = gradXest00 * mem[threadIdx.x*(blockDim.y+2)*input_zSize + threadIdx.y*input_zSize + threadIdx.z] +
				   gradXest01 * mem[threadIdx.x*(blockDim.y+2)*input_zSize + (threadIdx.y+1)*input_zSize + threadIdx.z] +
				   gradXest02 * mem[threadIdx.x*(blockDim.y+2)*input_zSize + (threadIdx.y+2)*input_zSize + threadIdx.z];
		grad_a01 = gradXest00 * mem[(threadIdx.x+1)*(blockDim.y+2)*input_zSize + threadIdx.y*input_zSize + threadIdx.z] +
				   gradXest01 * mem[(threadIdx.x+1)*(blockDim.y+2)*input_zSize + (threadIdx.y+1)*input_zSize + threadIdx.z] +
				   gradXest02 * mem[(threadIdx.x+1)*(blockDim.y+2)*input_zSize + (threadIdx.y+2)*input_zSize + threadIdx.z];
		grad_a02 = gradXest00 * mem[(threadIdx.x+2)*(blockDim.y+2)*input_zSize + threadIdx.y*input_zSize + threadIdx.z] +
				   gradXest01 * mem[(threadIdx.x+2)*(blockDim.y+2)*input_zSize + (threadIdx.y+1)*input_zSize + threadIdx.z] +
				   gradXest02 * mem[(threadIdx.x+2)*(blockDim.y+2)*input_zSize + (threadIdx.y+2)*input_zSize + threadIdx.z];
		
		grad_a10 = gradXest10 * mem[threadIdx.x*(blockDim.y+2)*input_zSize + threadIdx.y*input_zSize + threadIdx.z] +
				   gradXest11 * mem[threadIdx.x*(blockDim.y+2)*input_zSize + (threadIdx.y+1)*input_zSize + threadIdx.z] +
				   gradXest12 * mem[threadIdx.x*(blockDim.y+2)*input_zSize + (threadIdx.y+2)*input_zSize + threadIdx.z];
		grad_a11 = gradXest10 * mem[(threadIdx.x+1)*(blockDim.y+2)*input_zSize + threadIdx.y*input_zSize + threadIdx.z] +
				   gradXest11 * mem[(threadIdx.x+1)*(blockDim.y+2)*input_zSize + (threadIdx.y+1)*input_zSize + threadIdx.z] +
				   gradXest12 * mem[(threadIdx.x+1)*(blockDim.y+2)*input_zSize + (threadIdx.y+2)*input_zSize + threadIdx.z];
		grad_a12 = gradXest10 * mem[(threadIdx.x+2)*(blockDim.y+2)*input_zSize + threadIdx.y*input_zSize + threadIdx.z] +
				   gradXest11 * mem[(threadIdx.x+2)*(blockDim.y+2)*input_zSize + (threadIdx.y+1)*input_zSize + threadIdx.z] +
				   gradXest12 * mem[(threadIdx.x+2)*(blockDim.y+2)*input_zSize + (threadIdx.y+2)*input_zSize + threadIdx.z];
		
		grad_a20 = gradXest20 * mem[threadIdx.x*(blockDim.y+2)*input_zSize + threadIdx.y*input_zSize + threadIdx.z] +
				   gradXest21 * mem[threadIdx.x*(blockDim.y+2)*input_zSize + (threadIdx.y+1)*input_zSize + threadIdx.z] +
				   gradXest22 * mem[threadIdx.x*(blockDim.y+2)*input_zSize + (threadIdx.y+2)*input_zSize + threadIdx.z];
		grad_a21 = gradXest20 * mem[(threadIdx.x+1)*(blockDim.y+2)*input_zSize + threadIdx.y*input_zSize + threadIdx.z] +
				   gradXest21 * mem[(threadIdx.x+1)*(blockDim.y+2)*input_zSize + (threadIdx.y+1)*input_zSize + threadIdx.z] +
				   gradXest22 * mem[(threadIdx.x+1)*(blockDim.y+2)*input_zSize + (threadIdx.y+2)*input_zSize + threadIdx.z];
		grad_a22 = gradXest20 * mem[(threadIdx.x+2)*(blockDim.y+2)*input_zSize + threadIdx.y*input_zSize + threadIdx.z] +
				   gradXest21 * mem[(threadIdx.x+2)*(blockDim.y+2)*input_zSize + (threadIdx.y+1)*input_zSize + threadIdx.z] +
				   gradXest22 * mem[(threadIdx.x+2)*(blockDim.y+2)*input_zSize + (threadIdx.y+2)*input_zSize + threadIdx.z];
	
		// accumulate gradients
		atomicAdd(grad_a, grad_a00);
		atomicAdd(grad_a+1, grad_a01);
		atomicAdd(grad_a+2, grad_a02);
		atomicAdd(grad_a+3, grad_a10);
		atomicAdd(grad_a+4, grad_a11);
		atomicAdd(grad_a+5, grad_a12);
		atomicAdd(grad_a+6, grad_a20);
		atomicAdd(grad_a+7, grad_a21);
		atomicAdd(grad_a+8, grad_a22);
		
	}
}

void AmulBackwardKernelLauncher(const float* input, const float* weight, const float* gradOutput, float* gradInput, float* gradWeight, int axis,
                                const int batchSize, const int input_xSize, const int input_ySize, const int input_zSize, const int numChannels) {
	
    //float maxThreads = 1024.0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float maxThreads = prop.maxThreadsPerBlock;
    
    ZeroGradInputKernel<<<1,32>>>(gradInput, batchSize*input_xSize*input_ySize*input_zSize*numChannels);

	cudaDeviceSynchronize();
                      
    ZeroGradWeightKernel<<<1,1>>>(gradWeight, 9);
	
	cudaDeviceSynchronize();
	
    if (axis==0){
        int n = int(sqrt(maxThreads/input_xSize));   
        int ny = min(n,input_ySize);
        int nz = min(n,input_zSize);
            
        dim3 blockSize(input_xSize, ny, nz);   
	
    	int blockY = (input_ySize+ny-1)/ny;
    	int blockZ = (input_zSize+nz-1)/nz;
            		
    	dim3 gridSize(batchSize*numChannels, blockY, blockZ);
    
        AmulBackwardInputKernel_x<<<gridSize, blockSize>>>(weight, gradOutput, gradInput, batchSize, input_xSize, input_ySize, input_zSize, numChannels);
	
    	cudaDeviceSynchronize();
	
		AmulBackwardWeightKernel_x<<<gridSize, blockSize, input_xSize*(ny+2)*(nz+2)*sizeof(float)>>>(input, gradOutput, gradWeight, batchSize, input_xSize, input_ySize, input_zSize, numChannels);    
    }
    else if (axis==1){
        int n = int(sqrt(maxThreads/input_ySize));   
        int nx = min(n,input_xSize);
        int nz = min(n,input_zSize);
            
        dim3 blockSize(nx, input_ySize, nz);   
	
    	int blockX = (input_xSize+nx-1)/nx;
    	int blockZ = (input_zSize+nz-1)/nz;
		
    	dim3 gridSize(batchSize*numChannels, blockX, blockZ);
        
        AmulBackwardInputKernel_y<<<gridSize, blockSize>>>(weight, gradOutput, gradInput, batchSize, input_xSize, input_ySize, input_zSize, numChannels);
	
    	cudaDeviceSynchronize();
	
		AmulBackwardWeightKernel_y<<<gridSize, blockSize, (nx+2)*input_ySize*(nz+2)*sizeof(float)>>>(input, gradOutput, gradWeight, batchSize, input_xSize, input_ySize, input_zSize, numChannels);
    }
    else if (axis==2){
        int n = int(sqrt(maxThreads/input_zSize));  
        int nx = min(n,input_xSize);
        int ny = min(n,input_ySize);
            
        dim3 blockSize(nx, ny, input_zSize);   
	
    	int blockX = (input_xSize+nx-1)/nx;
    	int blockY = (input_ySize+ny-1)/ny;
    		
    	dim3 gridSize(batchSize*numChannels, blockX, blockY);
        //float time1,time2;
//        cudaEvent_t start1, stop1, start2, stop2;
//        cudaEventCreate(&start1);
//        cudaEventCreate(&stop1);
//        cudaEventRecord(start1,0);          
        AmulBackwardInputKernel_z<<<gridSize, blockSize>>>(weight, gradOutput, gradInput, batchSize, input_xSize, input_ySize, input_zSize, numChannels);
	
    	cudaDeviceSynchronize();
                          
//        cudaEventRecord(stop1,0);
//        cudaEventSynchronize(stop1);                        
//        cudaEventElapsedTime(&time1, start1, stop1) ;
//        cudaEventCreate(&start2);
//        cudaEventCreate(&stop2);
//        cudaEventRecord(start2,0);
//	
		AmulBackwardWeightKernel_z<<<gridSize, blockSize, (nx+2)*(ny+2)*input_zSize*sizeof(float)>>>(input, gradOutput, gradWeight, batchSize, input_xSize, input_ySize, input_zSize, numChannels);
    
      //  cudaDeviceSynchronize();
                    
//        cudaEventRecord(stop2,0);
//        cudaEventSynchronize(stop2);                        
//        cudaEventElapsedTime(&time2, start2, stop2) ;
//        printf("Time1: %3.2f ms \t\t Time2: %3.2f ms \n", time1, time2);
    }
	
}

#endif