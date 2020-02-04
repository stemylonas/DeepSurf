
#include "stdio.h"

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

__global__ void AmulBackwardInputKernel(const float* weight, const float* gradOutput, float* gradInput, const int batchSize, 
                                        const int gradInput_xSize, const int gradInput_ySize, const int gradInput_zSize, const int numChannels) { 

   // int blockDim = 10;                                    
                    
	int b = blockIdx.x/numChannels;
	int ch = blockIdx.x - b * numChannels;
	
	int x = threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	
	int gradInputBatchStride = b * gradInput_xSize * gradInput_ySize * gradInput_zSize * numChannels;
	
	int gradOutput_xSize = (gradInput_xSize-2) * 3;
    int gradOutput_ySize = (gradInput_ySize-2) * 3;
    int gradOutput_zSize = (gradInput_zSize-2) * 3;
    
	int gradOutputBatchStride = b * gradOutput_xSize * gradOutput_ySize * gradOutput_zSize * numChannels;
		
	if (x < (gradInput_xSize-2) && y < (gradInput_ySize-2) && z < (gradInput_zSize-2)) {
		
        float grad_y[27], temp;                                                             
        int i,j,k,l;                                                     
		for (i=0; i<3; i++)
            for (j=0; j<3; j++)
        		for (k=0; k<3; k++)			
            		grad_y[9*i+3*j+k] = gradOutput[gradOutputBatchStride + ((3*x+i)*gradOutput_ySize*gradOutput_zSize + (3*y+j)*gradOutput_zSize + (3*z+k))*numChannels + ch];
        
        const float* weight_start;	
        float* gradInputAddr = gradInput + gradInputBatchStride + (x*gradInput_ySize*gradInput_zSize + y*gradInput_zSize + z)*numChannels + ch;			

        for (i=0; i<3; i++)
            for (j=0; j<3; j++)
        		for (k=0; k<3; k++){ 
        			weight_start = weight + 9*i + 3*j + k;
                   
                    temp = 0;
                    for(l=0; l<27; l++)
            			temp += *(weight_start+27*l) * grad_y[l];
        			 
                    atomicAdd(gradInputAddr+(i*gradInput_ySize*gradInput_zSize+j*gradInput_zSize+k)*numChannels, temp);
         		}							
	}


}


__global__ void ZeroGradInputKernel(float *gradInput, int N){
	
	for (int i=threadIdx.x; i<N; i+=blockDim.x){
		gradInput[i] = 0.0f;
	}
	
}

__global__ void ZeroGradWeightKernel(float *gradWeight, float *gradWblocks, int w_size, int blocks){
	
    gradWeight[threadIdx.x] = 0.0f;
	
    for (int i=0; i<blocks; i++)
        gradWblocks[threadIdx.x+i*w_size] = 0.0f;
	
}
    
__global__ void AggregationWeightKernel(float *gradWeight, float* gradWblocks, int w_size, int nBlocks){
	
	for (int i=0; i<nBlocks; i++)
		gradWeight[threadIdx.x] += gradWblocks[i*w_size+threadIdx.x];
	
}

__global__ void AmulBackwardWeightKernel(const float *input, const float *gradOutput, float *grad_a, const int batchSize, 
                                         const int input_xSize, const int input_ySize, const int input_zSize, const int numChannels, const int mem_blocks){
	
    //int blockDim = 10;                                     
    //clock_t t1 = clock();         
                            
	int b = blockIdx.x/numChannels;
	int ch = blockIdx.x - b * numChannels;
	
	int x = threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
    
	
	extern __shared__ float mem_dw[];
    int mem_size = blockDim.x * (blockDim.y+2) * (blockDim.z+2);
	int inputBatchStride = b * input_xSize * input_ySize * input_zSize * numChannels;	
	
	int gradOutput_xSize = (input_xSize-2) * 3;
    int gradOutput_ySize = (input_ySize-2) * 3;
	int gradOutput_zSize = (input_zSize-2) * 3;
    
	int gradOutputBatchStride = b * gradOutput_xSize * gradOutput_ySize * gradOutput_zSize * numChannels;
	clock_t t2 = clock();
	for (int j=0; j<blockDim.y+2; j+=blockDim.y)
		for (int k=0; k<blockDim.z+2; k+=blockDim.z)
			if (j+threadIdx.y<blockDim.y+2 && k+threadIdx.z<blockDim.z+2 && (y+j)<input_ySize && (z+k)<input_zSize)
				mem_dw[threadIdx.x*(blockDim.y+2)*(blockDim.z+2) + (threadIdx.y+j)*(blockDim.z+2) + threadIdx.z+k] =
					input[inputBatchStride + (x*input_ySize*input_zSize + (y+j)*input_zSize+z+k)*numChannels + ch];
	//__syncthreads();

    //float *block_dw = mem_dw+mem_size;

    int nThreads = blockDim.x * blockDim.y * blockDim.z;
    //int nActiveThreads = (blockDim.x-2) * blockDim.y * blockDim.z;
    int thread_idx = threadIdx.x*blockDim.y*blockDim.z + threadIdx.y*blockDim.z + threadIdx.z;
    int block_idx = blockIdx.x*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + blockIdx.z;
    
    int w_size = 27*27;
   // for (int i=thread_idx; i<mem_blocks*w_size; i+=nThreads){
     //   block_dw[i] = 0.0;
        //printf("D");
    //}
    //printf("E"); 
    __syncthreads();
                      
	if (x < (input_xSize-2) && y < (input_ySize-2) && z < (input_zSize-2) ) {
    	//int active_thread_idx = threadIdx.x*blockDim.y*blockDim.z + threadIdx.y*blockDim.z + threadIdx.z;
		int gradXestStride = gradOutputBatchStride + (3*x*gradOutput_ySize*gradOutput_zSize + 3*y*gradOutput_zSize + 3*z)*numChannels + ch;
		int idx;	
		float y_grad[27], x_in[27], grad;
  		#pragma unroll
		for(int i=0; i<3; i++){
            #pragma unroll
            for(int j=0; j<3; j++){
                #pragma unroll
                for(int k=0; k<3; k++){
                    idx = 9*i+3*j+k;
                    y_grad[idx] = gradOutput[gradXestStride+(i*gradOutput_ySize*gradOutput_zSize+j*gradOutput_zSize+k)*numChannels];
                    x_in[idx] = mem_dw[(threadIdx.x+i)*(blockDim.y+2)*(blockDim.z+2) + (threadIdx.y+j)*(blockDim.z+2) + threadIdx.z+k];
                  //  block_dw[active_thread_idx*54+idx] = y_grad[idx]
                    //block_dw[active_thread_idx*54+27+idx] = x_in[idx]
                }
            }
        }
     
        idx = thread_idx % mem_blocks;
        float *grad_a_block = grad_a+idx*w_size;
        clock_t t4 = clock();
        #pragma unroll
        for(int i=0; i<27; i++){
            #pragma unroll
            for(int j=0; j<27; j++){
                //clock_t t3 = clock();    
                grad = y_grad[i]*x_in[j];
                //block_dw[thread_idx*w_size+27*i+j] = grad;
                //clock_t t4 = clock();
               // if (thread_idx<500){
                   // clock_t t4 = clock();
                //atomicAdd(&block_dw[idx+27*i+j],grad);
                atomicAdd(grad_a_block+27*i+j, grad);
                    //clock_t t5 = clock();
                    //printf("Atomic_add: %d\n", int(t5-t4));
                //}
                
    		}
        }
	    clock_t t5 = clock(); 
        //printf("Atomic_add (%) : %f \n", float(t5-t4)/int(t5-t2));
    }
    //clock_t t3 = clock();
   // __syncthreads();
                 
    //  
    
    
//    for (int j=thread_idx; j<w_size; j+=nThreads){
//        int r = j%27, pp = j/27;
//        float sum = 0.0;           
//        for (int i=0; i<nActiveThreads; i++)        
//            sum += block_dw[pp+i*54]*block_dw[27+r+i*54];  
//        grad_a[block_idx*w_size+27*pp+r] = sum;
//    }
   // printf("blockIdx=%d\n",block_idx);
//    for (int j=thread_idx; j<w_size; j+=nThreads){
//        float sum = 0.0;           
//        for (int i=0; i<mem_blocks; i++)        
//            sum += block_dw[j+i*w_size];  
//        grad_a[block_idx*w_size+j] = sum;
//    }
//   
   //  int block_idx = blockIdx.x*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + blockIdx.z;
    //for(int i=thread_idx; i<27*27; i+=nThreads)
      //  if (thread_idx<27*27) 
        //    grad_a[block_idx*27*27+i] = block_dw[i];
    //printf("Memory time : %d \n", int(t2-t1));
    //printf("Execution time : %d \n", int(t3-t2));
}


void AmulBackwardKernelLauncher(const float* input, const float* weight, const float* gradOutput, float* gradInput, float* gradWeight,
                                const int batchSize, const int input_xSize, const int input_ySize, const int input_zSize, const int numChannels) {
	
    float maxThreads = 128.0;
    //cudaDeviceProp prop;
    //cudaGetDeviceProperties(&prop, 0);
    //float maxThreads = prop.maxThreadsPerBlock;
    int n = int(sqrt(maxThreads/input_xSize)); 
    int ny = min(n,input_ySize);
    int nz = min(n,input_zSize);
    
    float time1,time2;
    cudaEvent_t start1, stop1, start2, stop2;
                
    ZeroGradInputKernel<<<1,maxThreads>>>(gradInput, batchSize*input_xSize*input_ySize*input_zSize*numChannels);

	cudaDeviceSynchronize();
                    
    float *gradWblocks;
    int k = maxThreads;
    cudaMalloc(&gradWblocks, k*27*27*sizeof(float)); 
                      
    ZeroGradWeightKernel<<<1,27*27>>>(gradWeight, gradWblocks, 27*27, k);
	
	cudaDeviceSynchronize();

    dim3 blockSize(input_xSize, ny, nz);   
	
	int blockY = (input_ySize+ny-1)/ny;
	int blockZ = (input_zSize+nz-1)/nz;
        		
	dim3 gridSize(batchSize*numChannels, blockY, blockZ);

    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1,0);
    AmulBackwardInputKernel<<<gridSize, blockSize>>>(weight, gradOutput, gradInput, batchSize, input_xSize, input_ySize, input_zSize, numChannels);
	                       
	cudaDeviceSynchronize();
                      
    cudaEventRecord(stop1,0);
    cudaEventSynchronize(stop1);                        
    cudaEventElapsedTime(&time1, start1, stop1) ;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2,0);
	int nblocks = batchSize*numChannels*blockY*blockZ;
    //float gradWblocks[27*27*nblocks];
    
         
   // int maxSharedMem = prop.sharedMemPerBlock;
    //int k = (maxSharedMem/sizeof(float)-input_xSize*(ny+2)*(nz+2))/(27*27);
    
   // printf("k=%d\n",k);
    //if ((27*27*k+input_xSize*(ny+2)*(nz+2))*sizeof(float)>maxSharedMem)
      //  printf("dsaasa\n\n");
	AmulBackwardWeightKernel<<<gridSize, blockSize, input_xSize*(ny+2)*(nz+2)*sizeof(float)>>>(input, gradOutput, gradWblocks, batchSize, input_xSize, input_ySize, input_zSize, numChannels, k);    
	
    cudaDeviceSynchronize();
                         
    cudaEventRecord(stop2,0);
    cudaEventSynchronize(stop2);                        
    cudaEventElapsedTime(&time2, start2, stop2) ;
  //  printf("Time1: %3.2f ms \t\t Time2: %3.2f ms, Dims (%dx%dx%d), Blocks %d \n", time1, time2, input_xSize, ny, nz, nblocks);
    AggregationWeightKernel<<<1,27*27>>>(gradWeight, gradWblocks, 27*27, k);   
    cudaFree(gradWblocks);
}

#endif
