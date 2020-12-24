#include "Variable.cuh"

OP::Variable::Variable(unsigned int minValue, unsigned int maxValue) :
    minValue(static_cast<uint8_t>(minValue)),
    maxValue(static_cast<uint8_t>(maxValue))
{}

unsigned int OP::Variable::cardinality() const
{
    return maxValue - minValue + 1u;
}

__host__ __device__
void OP::Variable::print()
{
    printf("(%d,%d)", minValue, maxValue);
}
