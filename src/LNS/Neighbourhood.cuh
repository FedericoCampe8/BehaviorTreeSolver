#pragma once

#include <algorithm>
#include <random>
#include <thrust/fill.h>
#include <Containers/Array.cuh>
#include <Containers/BitSet.cuh>
#include <OP/Problem.cuh>

class Neighbourhood
{
    // Aliases, Enums, ...
    public:
    enum ConstraintType: u8 {None, Eq, Neq};

    // Members
    public:
    float eqProbability;
    float neqProbability;
    Array<OP::ValueType> values;
    Array<ConstraintType> constraints;
    BitSet fixedValue;

    // Functions
    public:
    Neighbourhood(OP::Problem const * problem, float eqProbability, float neqProbability, Memory::MallocType mallocType);
    __host__ __device__ bool constraintsCheck(u32 variableIdx, OP::ValueType value) const;
    __host__ __device__ void reset();
    __host__ __device__ void constraintVariable(u32 variableIdx, OP::ValueType value, float random);
    void print(bool endLine = true);
};

Neighbourhood::Neighbourhood(OP::Problem const * problem,float eqProbability, float neqProbability, Memory::MallocType mallocType) :
    eqProbability(eqProbability),
    neqProbability(neqProbability),
    values(problem->variables.getCapacity(), mallocType),
    constraints(problem->variables.getCapacity(), mallocType),
    fixedValue(problem->maxValue + 1, mallocType)
{
    reset();
}

__host__ __device__
void Neighbourhood::reset()
{
    thrust::fill(thrust::seq, constraints.begin(), constraints.end(), ConstraintType::None);
    fixedValue.clear();
}

__host__ __device__
bool Neighbourhood::constraintsCheck(unsigned int variableIdx, OP::ValueType value) const
{
    if (*constraints[variableIdx] == ConstraintType::Eq and *values[variableIdx] != value)
    {
        return false;
    }
    else if(*constraints[variableIdx] == ConstraintType::Neq and *values[variableIdx] == value)
    {
        return false;
    }
    else if(*constraints[variableIdx] == ConstraintType::None and fixedValue.contains(value))
    {
        return false;
    }
    return true;
}

__host__ __device__
void Neighbourhood::constraintVariable(u32 variableIdx, OP::ValueType value, float random)
{
    if (random < eqProbability)
    {
        *constraints[variableIdx] = ConstraintType::Eq;
        *values[variableIdx] = value;
        fixedValue.insert(value);
    }
    else if (random < eqProbability + neqProbability)
    {
        *constraints[variableIdx] = ConstraintType::Neq;
        *values[variableIdx] = value;
    }
}

void Neighbourhood::print(bool endLine)
{
    auto printConstraint = [&](u32 variableIdx) -> void
    {
        switch (*constraints[variableIdx])
        {
            case ConstraintType::None:
                printf("*");
                break;
            case ConstraintType::Eq:
                printf("\033[30;42m%2d\033[0m", *values[variableIdx]);
                break;
            case ConstraintType::Neq:
                printf("\033[30;41m%2d\033[0m", *values[variableIdx]);
                break;
        }
    };

    printf("[");
    printConstraint(0);
    for (u32 variableIdx = 1; variableIdx < constraints.getCapacity(); variableIdx += 1)
    {
        printf(",");
        printConstraint(variableIdx);
    }
    printf(endLine ? "]\n" : "]");
}
