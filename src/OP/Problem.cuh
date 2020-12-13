#pragma once

#include <cstddef>

#include <Containers/RuntimeArray.cuh>

#include "Variable.cuh"

namespace OP
{
    class Problem
    {
        public:
            RuntimeArray<Variable> vars;

        public:
            __host__ __device__ Problem(unsigned int varsCount, std::byte* storage);
            __device__ Problem& operator=(Problem const & other);
            __host__ __device__ static std::size_t sizeOfStorage(unsigned int varsCount);

    };
}

