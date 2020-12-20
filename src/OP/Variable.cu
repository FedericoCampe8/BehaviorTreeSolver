#include "Variable.cuh"

OP::Variable::Variable(unsigned int minValue, unsigned int maxValue) :
    minValue(static_cast<uint8_t>(minValue)),
    maxValue(static_cast<uint8_t>(maxValue))
{}

unsigned int OP::Variable::cardinality(Variable const & variable)
{
    return variable.maxValue - variable.minValue + 1u;
}
