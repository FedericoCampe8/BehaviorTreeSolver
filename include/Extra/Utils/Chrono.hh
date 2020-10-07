#pragma once

#include <cstdint>
#include <chrono>

#include <Extra/Utils/Platform.hh>

namespace Extra::Utils::Chrono
{
    __host__ __device__
    inline uint64_t now()
    {
#ifdef __CUDA_ARCH__
        uint64_t ns;
        asm volatile("mov.u64 %0, %globaltimer;" : "=l"(ns));
        return ns / 1000000;
#else
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
#endif
    }
}