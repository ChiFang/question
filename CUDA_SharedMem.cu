#define USE_SHARED_MEM 1

__global__ void CalCostSmoothR_1(const int a_RangeUpScale, const int *a_CostData, int *a_Input)
{
	// Get the work index of the current element to be processed
	int y = blockIdx.x*blockDim.x + threadIdx.x;              //執行緒在陣列中對應的位置

#if USE_SHARED_MEM == 1
	__shared__ int Buff[32];
#else
	int Buff[32];
#endif

	// Do the operation
	for (int x = 1; (x < g_ImgWidth_CUDA); x++)
	{
		int TmpPos = y*Area + (x-1)*a_RangeUpScale;
#if USE_SHARED_MEM == 1
		// Synchronize to make sure the sub-matrices are loaded before starting the computation 
		__syncthreads();

		if (threadIdx.x < 32)
		{
			Buff[threadIdx.x] = a_CostSmooth[TmpPos + threadIdx.x];
		}

		// Synchronize to make sure the sub-matrices are loaded before starting the computation 
		__syncthreads();
#else
		for (int cnt = 0; cnt < 32 ;cnt++)
		{
			Buff[cnt] = a_CostSmooth[TmpPos + cnt];
		}
#endif

        // use Buff to do something
	}
}