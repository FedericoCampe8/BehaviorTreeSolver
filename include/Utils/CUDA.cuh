#ifdef __CUDA_ARCH__
#define CUDA_THREADS_BARRIER __syncthreads();
#define CUDA_ONLY_FIRST_THREAD if (threadIdx.x == 0)
#else
#define CUDA_THREADS_BARRIER
#define CUDA_ONLY_FIRST_THREAD
#endif