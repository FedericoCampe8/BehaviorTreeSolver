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
            Array<bool> fixedValues;
            Array<bool> fixedVariables;
            Array<unsigned int> fixedVariablesValues;

        // Functions
        public:
            Problem(unsigned int variablesCount, Memory::MallocType mallocType);
            void fixVariableWithValue(unsigned int variableIdx, unsigned int value);
            void unfixValue(unsigned int value);
    };
}

