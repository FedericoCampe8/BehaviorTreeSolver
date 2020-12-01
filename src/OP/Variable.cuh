#pragma once

namespace OP
{
    class Variable
    {
        public:
            int minValue;
            int maxValue;

        public:
            __host__ Variable(int minValue, int maxValue);
            __device__ Variable& operator=(Variable const & other);
            __host__ __device__ static unsigned int cardinality(Variable const & var);
    };
}