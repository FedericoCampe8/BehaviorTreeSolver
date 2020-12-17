#include "Variable.cuh"

OP::Variable::Variable(unsigned int minValue, unsigned int maxValue) :
    minValue(minValue),
    maxValue(maxValue)
{}

unsigned int OP::Variable::cardinality(Variable const & variable)
{
    return variable.maxValue - variable.minValue + 1;
}
