#pragma once

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
            ValueType calcMaxValue() const;
    };
}


