#pragma once

#include <cstddef>

#include <Containers/RuntimeArray.cuh>

#include "Variable.cuh"

namespace OP
{
    class Problem
    {
        public:
            RuntimeArray<Variable> variables;

        public:
            Problem(unsigned int variablesCount, std::byte* storage);
            std::byte* storageEnd() const;
            static std::size_t sizeOfStorage(unsigned int variablesCount);
    };
}

