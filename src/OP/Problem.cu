#include <algorithm>
#include <new>

#include "Problem.cuh"

__host__
OP::Problem::Problem(unsigned int varsCount, std::byte* storage):
    vars(varsCount, storage)
{}

__host__ __device__
std::size_t OP::Problem::sizeofStorage(unsigned int varsCount)
{
    return RuntimeArray<Variable>::sizeofStorage(varsCount);
}

__device__
OP::Problem& OP::Problem::operator=(Problem const & other)
{
    this->vars = other.vars;
    return *this;
}
