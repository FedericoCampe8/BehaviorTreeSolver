#pragma once

#include <random>
#include <curand_kernel.h>
#include <Utils/Memory.cuh>

class RandomEngine
{
    private:
    union Engine
    {
        std::mt19937 host;
        curandState device;
    };

    private:
    Engine engine;

    public:
    __host__ __device__ void initialize(u32 randomSeed);
    __host__ __device__ inline float getFloat01();
};

__host__ __device__
void RandomEngine::initialize(u32 randomSeed)
{
#ifdef __CUDA_ARCH__
    curand_init(randomSeed, 0, 0, &engine.device);
#else
    new (&engine.host) std::mt19937(randomSeed);
#endif
}

__host__ __device__
float RandomEngine::getFloat01()
{
#ifdef __CUDA_ARCH__
    return curand_uniform(&engine.device);
#else
    std::uniform_real_distribution<float> urd(0.0,1.0);
    return urd(engine.host);
#endif
}