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
        ValueType calcMaxValue() const;
    };

    template<typename ProblemType>
    ProblemType* parseInstance(char const * problemFilename, Memory::MallocType mallocType);
}

OP::Problem::Problem(unsigned int variablesCount, Memory::MallocType mallocType) :
    variables(variablesCount, mallocType)
{}

OP::ValueType OP::Problem::calcMaxValue() const
{
    ValueType maxValue = 0;
    for (Variable const * variable = variables.begin(); variable != variables.end(); variable += 1)
    {
        maxValue = max(maxValue, variable->maxValue);
    }
    return maxValue;
}



