#pragma once

class Variable
{
    public:
        int const minValue ;
        int const maxValue;

    public:
        Variable(int minValue, int maxValue);
        __host__ __device__ static unsigned int cardinality(Variable const & v);
};

