#pragma once

#include <cstddef>
#include <Containers/Array.cuh>

#include "Variable.cuh"

namespace OP
{
    class Problem
    {
        public:
            Array<Variable> variables;

        public:
            Problem(unsigned int variablesCount, Memory::MallocType mallocType);
    };
}

