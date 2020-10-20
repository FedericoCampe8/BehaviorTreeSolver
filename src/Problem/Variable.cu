#include <Problem/Variable.cuh>

Variable::Variable(int minValue, int maxValue) :
    minValue(minValue),
    maxValue(maxValue)
{}

__host__ __device__
unsigned int Variable::cardinality(Variable const & v)
{
    return v.maxValue - v.minValue + 1;
}
