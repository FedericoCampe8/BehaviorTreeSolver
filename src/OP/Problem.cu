#include "Problem.cuh"

OP::Problem::Problem(unsigned int variablesCount, Memory::MallocType mallocType) :
    variables(variablesCount, mallocType)
{}