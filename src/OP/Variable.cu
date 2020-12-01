#include "Variable.cuh"

__host__
OP::Variable::Variable(int minValue, int maxValue) :
    minValue(minValue),
    maxValue(maxValue)
{}

__host__ __device__
unsigned int OP::Variable::cardinality(Variable const & var)
{
    return var.maxValue - var.minValue + 1;
}

__device__
OP::Variable& OP::Variable::operator=(Variable const & other)
{
    this->minValue = other.minValue;
    this->maxValue = other.maxValue;
    return *this;
}
