#pragma once

#include <Containers/Array.cuh>

#include "Attributes.cuh"
#include "../OP/Problem.cuh"

namespace TS
{
    class Neighbourhood
    {
        // Members
        public:
            unsigned int tabuLength;
            int timestamp;
        private:
            unsigned int valuesCount;
            Array<TS::Attributes> attributes;

        // Functions
        public:
            Neighbourhood(OP::Problem const* problem, unsigned int tabuLength, Memory::MallocType mallocType);
            __host__ __device__ TS::Attributes* getAttributes(unsigned int variableIdx, unsigned int value) const;
            void operator=(Neighbourhood const & other);
            void update(LightVector<OP::Variable::ValueType> const * solution);
            void reset();
    };
}