#pragma once

#include <thrust/find.h>
#include <Containers/BitSet.cuh>
#include <Containers/Vector.cuh>
#include "../OP/Problem.cuh"
#include "Context.h"

namespace DP
{
    class State
    {
        // Members
        public:
        CostType cost;
        BitSet selectedValuesMap;
        BitSet admissibleValuesMap;
        Vector<OP::ValueType> selectedValues;

        // Functions
        public:
        __host__ __device__ State(OP::Problem const * problem, std::byte* storage);
        __host__ __device__ State(OP::Problem const * problem, Memory::MallocType mallocType);
        __host__ __device__ inline std::byte* endOfStorage() const;
        __host__ __device__ static std::byte* mallocStorages(OP::Problem const*  problem, u32 statesCount, Memory::MallocType mallocType);
        __host__ __device__ State& operator=(State const & other);
        __host__ __device__ inline void makeInvalid();
        __host__ __device__ void print(bool endLine = true) const;
        __host__ __device__ inline void selectValue(OP::ValueType value);
        __host__ __device__ static u32 sizeOfStorage(OP::Problem const* problem);
        __host__ __device__ static void swap(State& s0, State& s1);

    };
}

__host__ __device__
DP::State::State(OP::Problem const* problem, std::byte* storage) :
    cost(0),
    selectedValuesMap(problem->maxValue, reinterpret_cast<u32*>(storage)),
    admissibleValuesMap(problem->maxValue, Memory::align<u32>(selectedValuesMap.endOfStorage())),
    selectedValues(problem->variables.getCapacity(), Memory::align<OP::ValueType>(admissibleValuesMap.endOfStorage()))
{}

__host__ __device__
DP::State::State(OP::Problem const* problem, Memory::MallocType mallocType) :
    State(problem, mallocStorages(problem,1,mallocType))
{}

__host__ __device__
std::byte* DP::State::endOfStorage() const
{
    return selectedValues.endOfStorage();
}

__host__ __device__
std::byte* DP::State::mallocStorages(const OP::Problem* problem, u32 statesCount, Memory::MallocType mallocType)
{
    return Memory::safeMalloc(sizeOfStorage(problem) * statesCount, mallocType);
}

__host__ __device__
DP::State& DP::State::operator=(DP::State const & other)
{
    cost = other.cost;
    selectedValuesMap = other.selectedValuesMap;
    admissibleValuesMap = other.admissibleValuesMap;
    selectedValues = other.selectedValues;
    return *this;
}

__host__ __device__
void DP::State::makeInvalid()
{
    cost = DP::MaxCost;
}

__host__ __device__
void DP::State::print(bool endLine) const
{
    printf("Values: ");
    selectedValues.print(false);
    printf(" | Values map: ");
    selectedValuesMap.print(false);
    printf(" | Admissibles map: ");
    admissibleValuesMap.print(endLine);
}

__host__ __device__
void DP::State::selectValue(OP::ValueType value)
{
    selectedValues.pushBack(&value);
    selectedValuesMap.insert(value);
}

__host__ __device__
u32 DP::State::sizeOfStorage(OP::Problem const * problem)
{
    return
        BitSet::sizeOfStorage(problem->maxValue) + // selectedValuesMap
        BitSet::sizeOfStorage(problem->maxValue) + // admissibleValuesMap
        Vector<OP::ValueType>::sizeOfStorage(problem->variables.getCapacity()); // selectedValues
}

__host__ __device__
void DP::State::swap(DP::State& s0, DP::State& s1)
{
    thrust::swap(s0.cost, s1.cost);
    BitSet::swap(s0.selectedValuesMap, s1.selectedValuesMap);
    BitSet::swap(s0.admissibleValuesMap, s1.admissibleValuesMap);
    Vector<OP::ValueType>::swap(s0.selectedValues, s1.selectedValues);
}