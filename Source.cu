

#include<stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// includes CUDA
#include "cuda.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// nvcc does not seem to like variadic macros, so we have to define
// one for each kernel parameter list:
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

//索引用到的緒構體
struct Index {
	int block, thread;
};


//核心:把索引寫入裝置記憶體
__global__ void prob_idx(Index id[])
{
	int b = blockIdx.x;       //區塊索引
	int t = threadIdx.x;      //執行緒索引
	int n = blockDim.x;       //區塊中包含的執行緒數目
	int x = b*n + t;            //執行緒在陣列中對應的位置

								//每個執行緒寫入自己的區塊和執行緒索引.
	id[x].block = b;
	id[x].thread = t;
};

//主函式
int main() {
	Index* d;
	Index  h[100];

	//配置裝置記憶體
	cudaMalloc((void**)&d, 100 * sizeof(Index));

	//呼叫裝置核心
	int g = 3, b = 4, m = g*b;
	// prob_idx<<< g, b>>>(d);

	prob_idx KERNEL_ARGS2(dim3(nBlockCount), dim3(nThreadCount)) (d);

	//下載裝置記憶體內容到主機上
	cudaMemcpy(h, d, 100 * sizeof(Index), cudaMemcpyDeviceToHost);

	//顯示內容
	for (int i = 0; i<m; i++) {
		printf("h[%d]={block:%d, thread:%d}\n", i, h[i].block, h[i].thread);
	}

	//釋放裝置記憶體
	cudaFree(d);
	return 0;
}