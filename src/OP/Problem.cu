#include "Problem.cuh"

OP::Problem::Problem(unsigned int variablesCount, std::byte* storage) :
    variables(variablesCount, storage)
{}

std::byte* OP::Problem::storageEnd() const
{
    return variables.storageEnd();
}

std::size_t OP::Problem::sizeOfStorage(unsigned int variablesCount)
{
    return RuntimeArray<Variable>::sizeOfStorage(variablesCount);
}
