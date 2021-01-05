#include <thrust/fill.h>

#include "Neighbourhood.cuh"

LNS::Neighbourhood::Neighbourhood(OP::Problem const * problem, Memory::MallocType mallocType)  :
    constraints(problem->variables.getCapacity(), mallocType),
    solution(problem->variables.getCapacity(), mallocType),
    constrainedValues(problem->variables.getCapacity(), mallocType)
{
    thrust::fill(thrust::seq, constraints.begin(), constraints.end(), ConstraintType::None);
    thrust::fill(thrust::seq, constrainedValues.begin(), constrainedValues.end(), false);
}

void LNS::Neighbourhood::generate(LightArray<OP::Variable::ValueType> const * solution, unsigned int eqPercentage, unsigned int neqPercentage, std::mt19937* rng)
{
    this->solution = *solution;
    std::uniform_int_distribution<unsigned int> randomDistribution(0,100);
    for(unsigned int variablesIdx = 0; variablesIdx < solution->getCapacity(); variablesIdx += 1)
    {
        ConstraintType * const constraint = constraints[variablesIdx];
        *constraint = ConstraintType::None;
        OP::Variable::ValueType const value = *solution->at(variablesIdx);
        *constrainedValues[value] = false;
        unsigned int const random = randomDistribution(*rng);

        if (random < eqPercentage)
        {
            *constrainedValues[value] = true;
            *constraint = ConstraintType::Eq;
        }
        else if (random < eqPercentage + neqPercentage)
        {
            *constraint = ConstraintType::Neq;
        }
    }
}

void LNS::Neighbourhood::operator=(Neighbourhood const & other)
{
    constraints = other.constraints;
    solution = other.solution;
    constrainedValues = other.constrainedValues;
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
    printf("]");

    if(endLine)
    {
        printf("\n");
    }
}