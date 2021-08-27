#pragma once

#include <algorithm>
#include <thrust/fill.h>
#include <Containers/Array.cuh>
#include <Containers/BitSet.cuh>
#include <OP/Problem.cuh>
#include <Utils/Random.cuh>

class Neighbourhood
{
    // Aliases, Enums, ...
    private:
    enum ConstraintType: u8 {None, Eq, Neq};

    // Members
    private:
    float probEq;
    float probNeq;
    Array<OP::ValueType> values;
    Array<ConstraintType> constraints;
    BitSet fixedValue;

    // Functions
    public:
    Neighbourhood(OP::Problem const * problem, float probEq, float probNeq, Memory::MallocType mallocType);
    __host__ __device__ void generate(RandomEngine* randomEngine, Vector<OP::ValueType>* values);
    __host__ __device__ bool constraintsCheck(u32 variableIdx, OP::ValueType value) const;
    __host__ __device__ void clear();
    __host__ __device__ void constraintVariable(u32 variableIdx, OP::ValueType value, float random);
    void print(bool endLine = true);
};

Neighbourhood::Neighbourhood(OP::Problem const * problem,float probEq, float probNeq, Memory::MallocType mallocType) :
    probEq(probEq),
    probNeq(probNeq),
    values(problem->variables.getCapacity(), mallocType),
    constraints(problem->variables.getCapacity(), mallocType),
    fixedValue(problem->maxValue + 1, mallocType)
{
    clear();
}

__host__ __device__
void Neighbourhood::clear()
{
    thrust::fill(thrust::seq, constraints.begin(), constraints.end(), ConstraintType::None);
    fixedValue.clear();
}

__host__ __device__
void Neighbourhood::generate(RandomEngine* randomEngine, Vector<OP::ValueType>* values)
{
    clear();
    for(u32 valueIdx = 0; valueIdx < values->getCapacity(); valueIdx += 1)
    {
        OP::ValueType const value = *values->at(valueIdx);
        float const random = randomEngine->getFloat01();
        constraintVariable(valueIdx, value, random);
    }
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
    if (random < probEq)
    {
        *constraints[variableIdx] = ConstraintType::Eq;
        *values[variableIdx] = value;
        fixedValue.insert(value);
    }
    else if (random < probEq + probNeq)
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