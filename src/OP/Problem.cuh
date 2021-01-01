#pragma once

#include <cstddef>
#include <Containers/Array.cuh>

#include "Variable.cuh"

namespace OP
{
    class Problem
    {
        // Members
        public:
            Array<Variable> variables;

        // Functions
        public:
            Problem(unsigned int variablesCount, Memory::MallocType mallocType);
            unsigned int calcMaxOutdegree() const;
    };
}


