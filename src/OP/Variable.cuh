#pragma once

#include "Context.h"

namespace OP
{
    class Variable
    {
        // Members
        public:
        ValueType minValue;
        ValueType maxValue;

        // Functions
        public:
        Variable(ValueType minValue, ValueType maxValue);
        __host__ __device__ inline bool boundsCheck(ValueType value);
    };
}

OP::Variable::Variable(ValueType minValue, ValueType maxValue) :
    minValue(minValue),
    maxValue(maxValue)
{
    assert(maxValue < OP::MaxValue);
}

__host__ __device__
bool OP::Variable::boundsCheck(OP::ValueType value)
{
    return minValue <= value and value <= maxValue;
}
