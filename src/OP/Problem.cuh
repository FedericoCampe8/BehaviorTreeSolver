#pragma once

#include <Containers/Array.cuh>
#include "Variable.cuh"

namespace OP
{
    class Problem
    {
        // Members
        public:
        unsigned int maxBranchingFactor;
        Array<Variable> variables;

        // Functions
        public:
        Problem(unsigned int variablesCount, Memory::MallocType mallocType);
    };
}


