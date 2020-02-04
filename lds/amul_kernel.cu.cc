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



__global__ void AmulKernel_x(const float* input, const float* weight, float* output, const int batchSize, 
                           const int input_xSize, const int input_ySize, const int input_zSize, const int numChannels) {
	//int blockDim = 10;
                           
	int b = blockIdx.x/numChannels;
	int ch = blockIdx.x - b * numChannels;
	
	int x = threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z  = blockIdx.z * blockDim.z + threadIdx.z;
	//printf("%d %d %d",x,y,z);
	extern __shared__ float mem[]; 
	float x0, x1, x2;
	float x_temp0, x_temp1, x_temp2;
	
	float a00, a01, a02, a10, a11, a12, a20, a21, a22;  

    a00 = *weight;
	a01 = *(weight+1);
	a02 = *(weight+2);
	a10 = *(weight+3);
	a11 = *(weight+4);
	a12 = *(weight+5);
	a20 = *(weight+6);
	a21 = *(weight+7);
	a22 = *(weight+8);

    // case for different A across channels
//	a00 = *(weight+9*ch);
//	a01 = *(weight+9*ch+1);
//	a02 = *(weight+9*ch+2);
//	a10 = *(weight+9*ch+3);
//	a11 = *(weight+9*ch+4);
//	a12 = *(weight+9*ch+5);
//	a20 = *(weight+9*ch+6);
//	a21 = *(weight+9*ch+7);
//	a22 = *(weight+9*ch+8);
	
	int inputBatchStride = b * input_xSize * input_ySize * input_zSize * numChannels;	

	for (int j=0; j<blockDim.y+2; j+=blockDim.y)
		for (int k=0; k<blockDim.z+2; k+=blockDim.z)
			if (j+threadIdx.y<blockDim.y+2 && k+threadIdx.z<blockDim.z+2 && (y+j)<input_ySize && (z+k)<input_zSize)
				mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + (j+threadIdx.y)*(blockDim.z+2) + k+threadIdx.z] =
					input[inputBatchStride + (x*input_ySize*input_zSize + (y+j)*input_zSize+(z+k))*numChannels + ch];
	__syncthreads();
		
	
    int output_xSize = input_xSize;
    int output_ySize = (input_ySize-2) * 3;
    int output_zSize = (input_zSize-2) * 3;
	int outputBatchStride = b * output_xSize * output_ySize * output_zSize * numChannels;	
	// calculate x_est (calculate per column, save to output directly)
	
	if (x < input_xSize && y < (input_ySize-2) && z < (input_zSize-2))
    	//printf("get in");
		for (int k=0; k<3; k++){  // k == column
			
			x0 = mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + (threadIdx.z+k)];
			x1 = mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + (threadIdx.y+1)*(blockDim.z+2) + (threadIdx.z+k)];
			x2 = mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + (threadIdx.y+2)*(blockDim.z+2) + (threadIdx.z+k)];
			
			x_temp0 = a00 * x0 + a01 * x1 + a02 * x2;
			x_temp1 = a10 * x0 + a11 * x1 + a12 * x2;
			x_temp2 = a20 * x0 + a21 * x1 + a22 * x2;
			//printf("x %f\n",x_temp0);
			*(output + outputBatchStride + (x*output_ySize*output_zSize + (y*3)*output_zSize + (z*3+k))*numChannels + ch) = x_temp0;
			*(output + outputBatchStride + (x*output_ySize*output_zSize + (y*3+1)*output_zSize + (z*3+k))*numChannels + ch) = x_temp1;
			*(output + outputBatchStride + (x*output_ySize*output_zSize + (y*3+2)*output_zSize + (z*3+k))*numChannels + ch) = x_temp2;	
		}

}
        
__global__ void AmulKernel_y(const float* input, const float* weight, float* output, const int batchSize, 
                           const int input_xSize, const int input_ySize, const int input_zSize, const int numChannels) {
	//int blockDim = 10;
                           
	int b = blockIdx.x/numChannels;
	int ch = blockIdx.x - b * numChannels;
	
	int x = blockIdx.y * blockDim.x + threadIdx.x;
	int y = threadIdx.y;
    int z  = blockIdx.z * blockDim.z + threadIdx.z;
	
	extern __shared__ float mem[]; 
	float x0, x1, x2;
	float x_temp0, x_temp1, x_temp2;
	
	float a00, a01, a02, a10, a11, a12, a20, a21, a22;  
    
    a00 = *weight;
	a01 = *(weight+1);
	a02 = *(weight+2);
	a10 = *(weight+3);
	a11 = *(weight+4);
	a12 = *(weight+5);
	a20 = *(weight+6);
	a21 = *(weight+7);
	a22 = *(weight+8);

    // case for different A across channels
//	a00 = *(weight+9*ch);
//	a01 = *(weight+9*ch+1);
//	a02 = *(weight+9*ch+2);
//	a10 = *(weight+9*ch+3);
//	a11 = *(weight+9*ch+4);
//	a12 = *(weight+9*ch+5);
//	a20 = *(weight+9*ch+6);
//	a21 = *(weight+9*ch+7);
//	a22 = *(weight+9*ch+8);
	
	int inputBatchStride = b * input_xSize * input_ySize * input_zSize * numChannels;	

	for (int i=0; i<blockDim.x+2; i+=blockDim.x)
		for (int k=0; k<blockDim.z+2; k+=blockDim.z)
			if (i+threadIdx.x<blockDim.x+2 && k+threadIdx.z<blockDim.z+2 && (x+i)<input_xSize && (z+k)<input_zSize)
				mem[(i+threadIdx.x)*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + k+threadIdx.z] =
					input[inputBatchStride + ((x+i)*input_ySize*input_zSize + y*input_zSize+(z+k))*numChannels + ch];
	__syncthreads();
		
	
    int output_xSize = (input_xSize-2) * 3;
    int output_ySize = input_ySize;
    int output_zSize = (input_zSize-2) * 3;
	int outputBatchStride = b * output_xSize * output_ySize * output_zSize * numChannels;	
	// calculate x_est (calculate per column, save to output directly)
	
	if (x < (input_xSize-2) && y < input_ySize && z < (input_zSize-2))
	
		for (int k=0; k<3; k++){  // k == column
			
			x0 = mem[threadIdx.x*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + (threadIdx.z+k)];
			x1 = mem[(threadIdx.x+1)*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + (threadIdx.z+k)];
			x2 = mem[(threadIdx.x+2)*input_ySize*(blockDim.z+2) + threadIdx.y*(blockDim.z+2) + (threadIdx.z+k)];
			
			x_temp0 = a00 * x0 + a01 * x1 + a02 * x2;
			x_temp1 = a10 * x0 + a11 * x1 + a12 * x2;
			x_temp2 = a20 * x0 + a21 * x1 + a22 * x2;
			
			*(output + outputBatchStride + ((x*3)*output_ySize*output_zSize + y*output_zSize + (z*3+k))*numChannels + ch) = x_temp0;
			*(output + outputBatchStride + ((x*3+1)*output_ySize*output_zSize + y*output_zSize + (z*3+k))*numChannels + ch) = x_temp1;
			*(output + outputBatchStride + ((x*3+2)*output_ySize*output_zSize + y*output_zSize + (z*3+k))*numChannels + ch) = x_temp2;	
		}

}

__global__ void AmulKernel_z(const float* input, const float* weight, float* output, const int batchSize, 
                           const int input_xSize, const int input_ySize, const int input_zSize, const int numChannels) {
	//int blockDim = 12;
                           
	int b = blockIdx.x/numChannels;
	int ch = blockIdx.x - b * numChannels;
	
	int x = blockIdx.y * blockDim.x + threadIdx.x;
	int y = blockIdx.z * blockDim.y + threadIdx.y;
    int z  = threadIdx.z;
	
	extern __shared__ float mem[]; 
	float x0, x1, x2;
	float x_temp0, x_temp1, x_temp2;
	
	float a00, a01, a02, a10, a11, a12, a20, a21, a22;  

    a00 = *weight;
	a01 = *(weight+1);
	a02 = *(weight+2);
	a10 = *(weight+3);
	a11 = *(weight+4);
	a12 = *(weight+5);
	a20 = *(weight+6);
	a21 = *(weight+7);
	a22 = *(weight+8);

    // case for different A across channels
//	a00 = *(weight+9*ch);
//	a01 = *(weight+9*ch+1);
//	a02 = *(weight+9*ch+2);
//	a10 = *(weight+9*ch+3);
//	a11 = *(weight+9*ch+4);
//	a12 = *(weight+9*ch+5);
//	a20 = *(weight+9*ch+6);
//	a21 = *(weight+9*ch+7);
//	a22 = *(weight+9*ch+8);
	
	int inputBatchStride = b * input_xSize * input_ySize * input_zSize * numChannels;	

	for (int i=0; i<blockDim.x+2; i+=blockDim.x)
		for (int j=0; j<blockDim.y+2; j+=blockDim.y)
			if (i+threadIdx.x<blockDim.x+2 && j+threadIdx.y<blockDim.y+2 && (x+i)<input_xSize && (y+j)<input_ySize)
				mem[(i+threadIdx.x)*(blockDim.y+2)*input_zSize + (j+threadIdx.y)*input_zSize + threadIdx.z] =
					input[inputBatchStride + ((x+i)*input_ySize*input_zSize + (y+j)*input_zSize+z)*numChannels + ch];
	__syncthreads();
		
	
    int output_xSize = (input_xSize-2) * 3;
	int output_ySize = (input_ySize-2) * 3;
    int output_zSize = input_zSize;
	int outputBatchStride = b * output_xSize * output_ySize * output_zSize * numChannels;	
	// calculate x_est (calculate per column, save to output directly)
	
	if (x < (input_xSize-2) && y < (input_ySize-2) && z < input_zSize)
	
		for (int j=0; j<3; j++){  // j == column
			
			x0 = mem[threadIdx.x*(blockDim.y+2)*input_zSize + (threadIdx.y+j)*input_zSize + threadIdx.z];
			x1 = mem[(threadIdx.x+1)*(blockDim.y+2)*input_zSize + (threadIdx.y+j)*input_zSize + threadIdx.z];
			x2 = mem[(threadIdx.x+2)*(blockDim.y+2)*input_zSize + (threadIdx.y+j)*input_zSize + threadIdx.z];
			
			x_temp0 = a00 * x0 + a01 * x1 + a02 * x2;
			x_temp1 = a10 * x0 + a11 * x1 + a12 * x2;
			x_temp2 = a20 * x0 + a21 * x1 + a22 * x2;
			
			*(output + outputBatchStride + ((x*3)*output_ySize*output_zSize + (y*3+j)*output_zSize + z)*numChannels + ch) = x_temp0;
			*(output + outputBatchStride + ((x*3+1)*output_ySize*output_zSize + (y*3+j)*output_zSize + z)*numChannels + ch) = x_temp1;
			*(output + outputBatchStride + ((x*3+2)*output_ySize*output_zSize + (y*3+j)*output_zSize + z)*numChannels + ch) = x_temp2;	
		}

}

void AmulKernelLauncher(const float* input, const float* weight, float* output, int axis, const int batchSize,
                        const int input_xSize, const int input_ySize, const int input_zSize, const int numChannels) {
	
    //float maxThreads = 1024.0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float maxThreads = prop.maxThreadsPerBlock;

    if (axis==0){
        int n = int(sqrt(maxThreads/input_xSize));    
        int ny = min(n,input_ySize);
        int nz = min(n,input_zSize);
            
        dim3 blockSize(input_xSize, ny, nz);   
	
    	int blockY = (input_ySize+ny-1)/ny;
    	int blockZ = (input_zSize+nz-1)/nz;
            		
    	dim3 gridSize(batchSize*numChannels, blockY, blockZ);
    
    	AmulKernel_x<<<gridSize, blockSize, input_xSize*(ny+2)*(nz+2)*sizeof(float)>>>(input, weight, output, batchSize, input_xSize, input_ySize, input_zSize, numChannels);
        //std::cout << "end of calc";
    }
    else if (axis==1){
        int n = int(sqrt(maxThreads/input_ySize)); 
        int nx = min(n,input_xSize);
        int nz = min(n,input_zSize);
            
        dim3 blockSize(nx, input_ySize, nz);   
	
    	int blockX = (input_xSize+nx-1)/nx;
    	int blockZ = (input_zSize+nz-1)/nz;
		
    	dim3 gridSize(batchSize*numChannels, blockX, blockZ);

    	AmulKernel_y<<<gridSize, blockSize, (nx+2)*input_ySize*(nz+2)*sizeof(float)>>>(input, weight, output, batchSize, input_xSize, input_ySize, input_zSize, numChannels);
    }
    else if (axis==2){
        int n = int(sqrt(maxThreads/input_zSize)); 
        int nx = min(n,input_xSize);
        int ny = min(n,input_ySize);
            
        dim3 blockSize(nx, ny, input_zSize);   
	
    	int blockX = (input_xSize+nx-1)/nx;
    	int blockY = (input_ySize+ny-1)/ny;
    		
    	dim3 gridSize(batchSize*numChannels, blockX, blockY);
    
    	AmulKernel_z<<<gridSize, blockSize, (nx+2)*(ny+2)*input_zSize*sizeof(float)>>>(input, weight, output, batchSize, input_xSize, input_ySize, input_zSize, numChannels);         
       
        
        //if ( cudaSuccess != cudaGetLastError() )
         //   printf( "Error!\n" );
        //printf("esdde");
        //cudaDeviceSynchronize();
    }
	
}

#endif