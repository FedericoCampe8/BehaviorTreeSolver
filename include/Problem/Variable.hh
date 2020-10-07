#pragma once

#include <Extra/Extra.hh>

namespace Problem
{
    class Variable
    {
        public:
            int minValue;
            int maxValue;

        public:
            Variable(int minValue, int maxValue);
    };
}
