#include <thrust/fill.h>

#include "Neighbourhood.cuh"

LNS::Neighbourhood::Neighbourhood(OP::Problem const * problem, Memory::MallocType mallocType)  :
    fixedValues(problem->variables.getCapacity(), mallocType),
    fixedVariables(problem->variables.getCapacity(), mallocType),
    fixedVariablesValues(problem->variables.getCapacity(), mallocType)
{
    reset();
}

void LNS::Neighbourhood::fixVariables(LightArray<OP::Variable::ValueType> const * solution, unsigned int fixPercentage, std::mt19937* rng)
{
    std::uniform_int_distribution<unsigned int> randomDistribution(0,100);
    for(unsigned int variablesIdx = 0; variablesIdx < solution->getCapacity(); variablesIdx += 1)
    {
        unsigned int random = randomDistribution(*rng);
        registerVariableWithValue(random <= fixPercentage, variablesIdx, *solution->at(variablesIdx));
    }
}

void LNS::Neighbourhood::operator=(const Neighbourhood& other)
{
    fixedValues = other.fixedValues;
    fixedVariables = other.fixedVariables;
    fixedVariablesValues = other.fixedVariablesValues;
}

void LNS::Neighbourhood::print(bool endLine)
{
    printf("[");
    if(*fixedVariables[0])
    {
        printf("%2d", *fixedVariablesValues[0]);
    }
    else
    {
        printf("__");
    }
    for(unsigned int variableIdx = 1; variableIdx < fixedVariablesValues.getCapacity(); variableIdx += 1)
    {
        if(*fixedVariables[variableIdx])
        {
            printf(",%2d", *fixedVariablesValues[variableIdx]);
        }
        else
        {
            printf(",__");
        }
    }
    printf("]");

    if(endLine)
    {
        printf("\n");
    }
}


void LNS::Neighbourhood::reset()
{
    thrust::fill(thrust::seq, fixedValues.begin(), fixedValues.end(), false);
    thrust::fill(thrust::seq, fixedVariables.begin(), fixedVariables.end(), false);
}

void LNS::Neighbourhood::registerVariableWithValue(bool fixed, unsigned int variableIdx, unsigned int value)
{
    *fixedValues[value] = fixed;
    *fixedVariables[variableIdx] = fixed;
    *fixedVariablesValues[variableIdx] = value;
}