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



__global__ void AmulKernel(const float* input, const float* weight, float* output, const int batchSize, 
                           const int input_xSize, const int input_ySize, const int input_zSize, const int numChannels) {
	//int blockDim = 10;
                           
	int b = blockIdx.x/numChannels;
	int ch = blockIdx.x - b * numChannels;
	
	int x = threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z  = blockIdx.z * blockDim.z + threadIdx.z;
	//printf("%d %d %d",x,y,z);
	extern __shared__ float mem[]; 
	float y0;
	const float* weight_start;
	
	int inputBatchStride = b * input_xSize * input_ySize * input_zSize * numChannels;	
	for (int j=0; j<blockDim.y+2; j+=blockDim.y)
		for (int k=0; k<blockDim.z+2; k+=blockDim.z)
			if (j+threadIdx.y<blockDim.y+2 && k+threadIdx.z<blockDim.z+2 && (y+j)<input_ySize && (z+k)<input_zSize)
				mem[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + (j+threadIdx.y)*(blockDim.z+2) + k+threadIdx.z] =
					input[inputBatchStride + (x*input_ySize*input_zSize + (y+j)*input_zSize+(z+k))*numChannels + ch];
	__syncthreads();
		
	
    int output_xSize = (input_xSize-2) * 3;
    int output_ySize = (input_ySize-2) * 3;
    int output_zSize = (input_zSize-2) * 3;
	int outputBatchStride = b * output_xSize * output_ySize * output_zSize * numChannels;	
	
	if (x < (input_xSize-2) && y < (input_ySize-2) && z < (input_zSize-2)){
    
        float x_in[27];
                  
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
        		for (int k=0; k<3; k++)
        			x_in[9*i+3*j+k] = mem[(threadIdx.x+i)*(blockDim.y+2)*(blockDim.z+2) + (threadIdx.y+j)*(blockDim.z+2) + (threadIdx.z+k)];
                  
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
        		for (int k=0; k<3; k++){ 
        			weight_start = weight + 27*(9*i + 3*j + k);
                   
                    y0 = 0;
                    for(int l=0; l<27; l++)
            			y0 += *(weight_start+l) * x_in[l];
        			
        			*(output + outputBatchStride + ((x*3+i)*output_ySize*output_zSize + (y*3+j)*output_zSize + (z*3+k))*numChannels + ch) = y0;
        			
        		}
    }

}
        

void AmulKernelLauncher(const float* input, const float* weight, float* output, const int batchSize,
                        const int input_xSize, const int input_ySize, const int input_zSize, const int numChannels) {
	
    float maxThreads = 256.0;
  //  cudaDeviceProp prop;
   // cudaGetDeviceProperties(&prop, 0);
    //float maxThreads = prop.maxThreadsPerBlock;
    
    int n = int(sqrt(maxThreads/input_xSize)); 
    int ny = min(n,input_ySize);
    int nz = min(n,input_zSize);

    dim3 blockSize(input_xSize, ny, nz);   
	
	int blockY = (input_ySize+ny-1)/ny;
	int blockZ = (input_zSize+nz-1)/nz;
        		
	dim3 gridSize(batchSize*numChannels, blockY, blockZ);

	AmulKernel<<<gridSize, blockSize, input_xSize*(ny+2)*(nz+2)*sizeof(float)>>>(input, weight, output, batchSize, input_xSize, input_ySize, input_zSize, numChannels);
	
}

#endif
