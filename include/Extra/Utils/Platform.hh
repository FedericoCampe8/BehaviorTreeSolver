#pragma once

#include <cassert>
#include <cstdlib>

#ifdef GPU
    #define CALL(gridSize, blockSize, sharedMemorySize, cudaStream, fun, args...) fun<<<gridSize, blockSize, sharedMemorySize, cudaStream>>>(args);
#else
    #define __device__
    #define __host__
    #define __global__
    #define __managed__
    #define __shared__
    #define CALL(gridSize, blockSize, sharedMemorySize, cudaStream, fun, args...) fun(args);
#endif



namespace Extra::Utils::Platform
{
    __host__ __device__
    inline void abort()
    {
#ifdef __CUDA_ARCH__
        __trap();
#else
        std::abort();
#endif
    }

    #undef assert
    __host__ __device__
    inline void assert(bool b)
    {
#ifndef NDEBUG
       if(not b)
           abort();
#endif
    }
}

