#pragma once

#include <algorithm>
#include <random>
#include <thrust/fill.h>
#include <Containers/Array.cuh>
#include <Containers/BitSet.cuh>
#include "OP/Problem.cuh"

class Neighbourhood
{
    // Aliases, Enums, ...
    public:
    enum ConstraintType: u8 {None, Eq, Neq};

    // Members
    public:
    Vector<OP::ValueType> solution;
    Array<ConstraintType> constraints;
    BitSet fixedValue;

    // Functions
    public:
    Neighbourhood(OP::Problem const * problem, Memory::MallocType mallocType);
    __host__ __device__ bool constraintsCheck(u32 variableIdx, OP::ValueType value) const;
    void generate(Vector<OP::ValueType> const * solution, u32 eqPercentage, u32 neqPercentage, std::mt19937* rng);
    void print(bool endLine = true);
    void prefetchAsync(i32 dstDevice);
};

Neighbourhood::Neighbourhood(OP::Problem const * problem, Memory::MallocType mallocType) :
    solution(problem->variables.getCapacity(), mallocType),
    constraints(problem->variables.getCapacity(), mallocType),
    fixedValue(problem->maxValue + 1, mallocType)
{
    thrust::fill(thrust::seq, constraints.begin(), constraints.end(), ConstraintType::None);
    fixedValue.clear();
}

__host__ __device__
bool Neighbourhood::constraintsCheck(unsigned int variableIdx, OP::ValueType value) const
{
    if (*constraints[variableIdx] == ConstraintType::Eq and *solution[variableIdx] != value)
    {
        return false;
    }
    else if(*constraints[variableIdx] == ConstraintType::Neq and *solution[variableIdx] == value)
    {
        return false;
    }
    else if(*constraints[variableIdx] == ConstraintType::None and fixedValue.contains(value))
    {
        return false;
    }
    return true;
}

void Neighbourhood::generate(Vector<OP::ValueType> const * solution, u32 eqPercentage, u32 neqPercentage, std::mt19937* rng)
{
    this->solution = *solution;
    std::uniform_int_distribution<u32> randomDistribution(0, 100);
    for (u32 variableIdx = 0; variableIdx < solution->getCapacity(); variableIdx += 1)
    {
        ConstraintType* const constraint = constraints[variableIdx];
        OP::ValueType const value = *solution->at(variableIdx);
        *constraint = ConstraintType::None;
        fixedValue.erase(value);
        u32 random = randomDistribution(*rng);
        if (random < eqPercentage)
        {
            *constraint = ConstraintType::Eq;
            fixedValue.insert(value);
        }
        else if (random < eqPercentage + neqPercentage)
        {
            *constraint = ConstraintType::Neq;
        }
    }
}

void Neighbourhood::print(bool endLine)
{
    auto printConstraint = [&](u32 variableIdx) -> void
    {
        switch (*constraints[variableIdx])
        {
            case ConstraintType::None:
                printf("  ");
                break;
            case ConstraintType::Eq:
                printf("\033[30;42m%2d\033[0m", *solution[variableIdx]);
                break;
            case ConstraintType::Neq:
                printf("\033[30;41m%2d\033[0m", *solution[variableIdx]);
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


