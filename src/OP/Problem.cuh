#pragma once

#include <Containers/Vector.cuh>
#include "Variable.cuh"

namespace OP
{
    class Problem
    {
        // Members
        public:
        u32 maxValue;
        u32 maxBranchingFactor;
        Vector<Variable> variables;

        // Functions
        public:
        Problem(u32 variablesCount, Memory::MallocType mallocType);
        void add(Variable const * variable);
    };

    template<typename ProblemType>
    ProblemType* parseInstance(char const * problemFilename, Memory::MallocType mallocType);
}

OP::Problem::Problem(u32 variablesCount, Memory::MallocType mallocType) :
    maxValue(0),
    maxBranchingFactor(0),
    variables(variablesCount, mallocType)
{}

void OP::Problem::add(Variable const * variable)
{
    variables.pushBack(variable);
    maxValue = max(maxValue, variable->maxValue);
    maxBranchingFactor = maxValue + 1;
}



