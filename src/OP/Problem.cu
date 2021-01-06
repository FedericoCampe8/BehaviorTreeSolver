#include "Problem.cuh"

OP::Problem::Problem(unsigned int variablesCount, Memory::MallocType mallocType) :
    variables(variablesCount, mallocType)
{}

OP::ValueType OP::Problem::calcMaxValue() const
{
    ValueType maxValue = 0;
    for (OP::Variable const * variable = variables.begin(); variable != variables.end(); variable += 1)
    {
        maxValue = max(maxValue, variable->maxValue);
    }
    return maxValue;
}

