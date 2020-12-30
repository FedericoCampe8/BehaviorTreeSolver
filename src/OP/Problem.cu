#include "Problem.cuh"

OP::Problem::Problem(unsigned int variablesCount, Memory::MallocType mallocType) :
    variables(variablesCount, mallocType),
    fixedValues(variablesCount, mallocType),
    fixedVariables(variablesCount, mallocType),
    fixedVariablesValues(variablesCount, mallocType)
{
    thrust::fill(thrust::seq, fixedValues.begin(), fixedValues.end(), false);
    thrust::fill(thrust::seq, fixedVariables.begin(), fixedVariables.end(), false);
}

void OP::Problem::fixVariableWithValue(unsigned int variableIdx, unsigned int value)
{
    *fixedValues[value] = true;
    *fixedVariables[variableIdx] = true;
    *fixedVariablesValues[variableIdx] = value;
}

void OP::Problem::unfixValue(unsigned int value)
{
    *fixedValues[value] = false;
}
