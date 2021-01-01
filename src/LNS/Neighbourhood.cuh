#pragma once

#include <Containers/Array.cuh>

#include "../OP/Problem.cuh"

namespace LNS
{
    class Neighbourhood
    {
            // Members
        public:
            Array<bool> fixedValues;
            Array<bool> fixedVariables;
            Array<OP::Variable::ValueType> fixedVariablesValues;

            // Functions
        public:
            Neighbourhood(OP::Problem const* problem, Memory::MallocType mallocType);
            void fixVariables(LightArray<OP::Variable::ValueType> const* solution, unsigned int fixPercentage, unsigned int randomSeed);
            void operator=(Neighbourhood const& other);
            void print(bool endLine = true);
            void reset();

        private:
            void fixVariableWithValue(unsigned int variableIdx, unsigned int value);
    };
}
