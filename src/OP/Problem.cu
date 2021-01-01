#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include "Problem.cuh"

OP::Problem::Problem(unsigned int variablesCount, Memory::MallocType mallocType) :
    variables(variablesCount, mallocType)
{}

unsigned int OP::Problem::calcMaxOutdegree() const
{
    unsigned int maxOutdegree = 0;
    for (OP::Variable* variable = variables.begin(); variable != variables.end(); variable += 1)
    {
        maxOutdegree = max(maxOutdegree, variable->cardinality());
    }

    return maxOutdegree;
}

