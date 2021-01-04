#pragma once

#include <random>
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
            void fixVariables(LightArray<OP::Variable::ValueType> const * solution, unsigned int fixPercentage, std::mt19937* rng);
            void operator=(Neighbourhood const& other);
            void print(bool endLine = true);
            void reset();

        private:
            void registerVariableWithValue(bool fixed, unsigned int variableIdx, unsigned int value);
    };
}