#include "Variable.cuh"

OP::Variable::Variable(ValueType minValue, ValueType maxValue) :
    minValue(minValue),
    maxValue(maxValue)
{}

unsigned int OP::Variable::cardinality(Variable const & variable)
{
    return variable.maxValue - variable.minValue + 1;
}
