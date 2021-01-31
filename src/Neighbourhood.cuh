#pragma once

#include <algorithm>
#include <random>
#include <thrust/fill.h>
#include <Containers/Array.cuh>
#include "OP/Problem.cuh"

class Neighbourhood
{
    // Aliases, Enums, ...
    public:
    enum ConstraintType {None, Eq, Neq};

    // Members
    public:
    Array<ConstraintType> constraints;
    Array<OP::ValueType> solution;
    Array<bool> fixedValue;

    // Functions
    public:
    Neighbourhood(OP::Problem const * problem, Memory::MallocType mallocType);
    __host__ __device__ bool constraintsCheck(unsigned int variableIdx, OP::ValueType value) const;
    void generate(LightArray<OP::ValueType> const* solution, unsigned int eqPercentage, unsigned int neqPercentage, std::mt19937* rng);
    void print(bool endLine = true);
};

Neighbourhood::Neighbourhood(OP::Problem const * problem, Memory::MallocType mallocType) :
    constraints(problem->variables.getCapacity(), mallocType),
    solution(problem->variables.getCapacity(), mallocType),
    fixedValue(problem->calcMaxValue() + 1, mallocType)
{
    thrust::fill(thrust::seq, constraints.begin(), constraints.end(), ConstraintType::None);
    thrust::fill(thrust::seq, fixedValue.begin(), fixedValue.end(), false);
}

__host__ __device__
bool Neighbourhood::constraintsCheck(unsigned int variableIdx, OP::ValueType value) const
{
    Neighbourhood::ConstraintType const constraint = *constraints[variableIdx];
    bool const isFree = constraint == ConstraintType::None and (not *fixedValue[value]);
    bool const isEq = constraint == ConstraintType::Eq and *solution[variableIdx] == value;
    bool const isNeq = constraint == ConstraintType::Neq and *solution[variableIdx] != value;
    return isFree or isEq or isNeq;
}

void Neighbourhood::generate(LightArray<OP::ValueType> const * solution, unsigned int eqPercentage, unsigned int neqPercentage, std::mt19937* rng)
{
    std::uniform_int_distribution<unsigned int> randomDistribution(0, 100);
    for (unsigned int variableIdx = 0; variableIdx < solution->getCapacity(); variableIdx += 1)
    {
        ConstraintType* const constraint = constraints[variableIdx];
        *constraint = ConstraintType::None;
        OP::ValueType const value = *solution->at(variableIdx);
        *fixedValue[value] = false;
        unsigned int const random = randomDistribution(*rng);
        if (random < eqPercentage)
        {
            *fixedValue[value] = true;
            *constraint = ConstraintType::Eq;
            *this->solution[variableIdx] = value;
        }
        else if (random < eqPercentage + neqPercentage)
        {
            *constraint = ConstraintType::Neq;
            *this->solution[variableIdx] = value;
        }
    }
}

void Neighbourhood::print(bool endLine)
{
    auto printConstraint = [&](unsigned int variableIdx) -> void
    {
        switch (*constraints[variableIdx])
        {
            case ConstraintType::None:
                printf("  ");
                break;
            case ConstraintType::Eq:
                printf("\033[30;42m%2d\033[0m", * solution[variableIdx]);
                break;
            case ConstraintType::Neq:
                printf("\033[30;41m%2d\033[0m", * solution[variableIdx]);
                break;
        }
    };

    printf("[");
    printConstraint(0);
    for (unsigned int variableIdx = 1; variableIdx < constraints.getCapacity(); variableIdx += 1)
    {
        printf(",");
        printConstraint(variableIdx);
    }
    printf(endLine ? "]\n" : "]");
}


