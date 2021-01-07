#include "Problem.cuh"

OP::Problem::Problem(unsigned int variablesCount, Memory::MallocType mallocType) :
    maxBranchingFactor(0),
    variables(variablesCount, mallocType)
{}