#pragma once

#include <Problem/State.hh>
#include <Extra/Containers.hh>

class MDD;

namespace Problem
{
    struct AllDifferent
    {
        class State : public Problem::State
        {
            private:
                Extra::Containers::RestrainedVector<int> selectedValues;

            public:
                __device__ static std::size_t getSizeStorage(MDD const * const mdd);
                __device__ State(Type type, std::size_t sizeStorage, std::byte *const storage);
                __device__ State & operator=(State const &other);
                __device__ bool operator==(State const &other) const;
                __device__ void next(int value, State * const child) const;
            private:
                __device__ bool isValueSelected(int value) const;
                __device__ void addToSelectedValues(int value);
        };

        __host__ static uint getOptimalLayerWidth(uint variablesCount);
    };
}
