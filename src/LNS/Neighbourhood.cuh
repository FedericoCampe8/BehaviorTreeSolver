#pragma once

#include <random>
#include <Containers/Array.cuh>

#include "../OP/Problem.cuh"

namespace LNS
{
    class Neighbourhood
    {
            // Aliases, Enums, ...
        public:
            enum ConstraintType : uint8_t {None, Eq, Neq};

            // Members
        public:
            Array<ConstraintType> constraints;
            Array<OP::Variable::ValueType> solution;
            Array<bool> constrainedValues;

            // Functions
        public:
            Neighbourhood(OP::Problem const* problem, Memory::MallocType mallocType);
            void generate(LightArray<OP::Variable::ValueType> const * solution, unsigned int eqPercentage, unsigned int neqPercentage, std::mt19937* rng);
            void operator=(Neighbourhood const& other);
            void print(bool endLine = true);
    };
}