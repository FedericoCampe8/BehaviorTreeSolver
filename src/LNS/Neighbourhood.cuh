#pragma once

#include <random>
#include <Containers/Array.cuh>

#include "../OP/Problem.cuh"

namespace LNS
{
    enum ConstraintType : uint8_t {None, Eq, Neq};

    class Neighbourhood
    {
        // Members
        public:
            LightArray<ConstraintType> constraints;
            LightArray<OP::ValueType> solution;
            LightArray<bool> constrainedValues;

        // Functions
        public:
            Neighbourhood(OP::Problem const* problem, std::byte* storage);
            void generate(LightArray<OP::ValueType> const * solution, unsigned int eqPercentage, unsigned int neqPercentage, std::mt19937* rng);
            static std::byte* mallocStorages(OP::Problem const * problem, unsigned int neighbourhoodsCount, Memory::MallocType mallocType);
            void print(bool endLine = true);
            static unsigned int sizeOfStorage(OP::Problem const * problem);
    };
}