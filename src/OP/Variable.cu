#include "Variable.cuh"

OP::Variable::Variable(unsigned int minValue, unsigned int maxValue) :
    minValue(static_cast<uint8_t>(minValue)),
    maxValue(static_cast<uint8_t>(maxValue))
{}

unsigned int OP::Variable::cardinality() const
{
    return maxValue - minValue + 1u;
}

void OP::Variable::fixTo(unsigned int value)
{
    minValue = value;
    maxValue = value;
}

bool OP::Variable::isFixed() const
{
    return minValue == maxValue;
}

