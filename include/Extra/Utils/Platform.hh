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

