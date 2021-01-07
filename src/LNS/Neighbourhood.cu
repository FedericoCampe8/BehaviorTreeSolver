#include <thrust/fill.h>

#include "Neighbourhood.cuh"

LNS::Neighbourhood::Neighbourhood(OP::Problem const * problem, std::byte* storage)  :
    constraints(problem->variables.getCapacity(), reinterpret_cast<ConstraintType*>(storage)),
    solution(problem->variables.getCapacity(), reinterpret_cast<OP::ValueType*>(Memory::align(8u, constraints.end()))),
    constrainedValues(problem->variables.getCapacity(), reinterpret_cast<bool*>(Memory::align(8u, solution.end())))
{
    thrust::fill(thrust::seq, constraints.begin(), constraints.end(), ConstraintType::None);
    thrust::fill(thrust::seq, constrainedValues.begin(), constrainedValues.end(), false);
}

void LNS::Neighbourhood::generate(LightArray<OP::ValueType> const * solution, unsigned int eqPercentage, unsigned int neqPercentage, std::mt19937* rng)
{
    std::uniform_int_distribution<unsigned int> randomDistribution(0,100);
    for(unsigned int variablesIdx = 0; variablesIdx < solution->getCapacity(); variablesIdx += 1)
    {
        ConstraintType * const constraint = constraints[variablesIdx];
        *constraint = ConstraintType::None;
        OP::ValueType const value = *solution->at(variablesIdx);
        *constrainedValues[value] = false;
        unsigned int const random = randomDistribution(*rng);

        if (random < eqPercentage)
        {
            *constrainedValues[value] = true;
            *constraint = ConstraintType::Eq;
            *this->solution[variablesIdx] = *solution->at(variablesIdx);
        }
        else if (random < eqPercentage + neqPercentage)
        {
            *constraint = ConstraintType::Neq;
            *this->solution[variablesIdx] = *solution->at(variablesIdx);
        }
    }
}

std::byte* LNS::Neighbourhood::mallocStorages(const OP::Problem* problem, unsigned int neighbourhoodsCount, Memory::MallocType mallocType)
{
    return Memory::safeMalloc(sizeOfStorage(problem) * neighbourhoodsCount, mallocType);
}

void LNS::Neighbourhood::print(bool endLine)
{
    auto printConstraint = [&] (unsigned int variableIdx) -> void
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
    for(unsigned int variableIdx = 1; variableIdx < constraints.getCapacity(); variableIdx += 1)
    {
        printf(",");
        printConstraint(variableIdx);
    }
    printf("]%c", endLine ? '\n' : '\0');
}

unsigned int LNS::Neighbourhood::sizeOfStorage(OP::Problem const * problem)
{
    return
        LightArray<ConstraintType>::sizeOfStorage(problem->variables.getCapacity()) + // constraints
        LightArray<OP::ValueType>::sizeOfStorage(problem->variables.getCapacity()) + // solution
        LightArray<bool>::sizeOfStorage(problem->calcMaxValue()) + // constrainedValues
        8 * 3; // Alignment
}
