#pragma once

#include <thrust/find.h>
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
        Vector<OP::ValueType> selectedValues;
        Vector<OP::ValueType> admissibleValues;

        // Functions
        public:
        __host__ __device__ State(OP::Problem const * problem, std::byte* storage);
        __host__ __device__ State(OP::Problem const * problem, Memory::MallocType mallocType);
        __host__ __device__ bool isAdmissible(OP::ValueType value) const;
        __host__ __device__ static std::byte* mallocStorages(OP::Problem const*  problem, unsigned int statesCount, Memory::MallocType mallocType);
        __host__ __device__ State& operator=(State const & other);
        __host__ __device__ void removeFromAdmissibles(OP::ValueType value);
        __host__ __device__ void setInvalid();
        __host__ __device__ static unsigned int sizeOfStorage(OP::Problem const* problem);
        __host__ __device__ static void swap(State& s0, State& s1);

    };
}

__host__ __device__
DP::State::State(OP::Problem const* problem, std::byte* storage) :
    cost(0),
    selectedValues(problem->variables.getCapacity(), reinterpret_cast<OP::ValueType*>(storage)),
    admissibleValues(problem->variables.getCapacity(), reinterpret_cast<OP::ValueType*>(selectedValues.endOfStorage()))
{}

__host__ __device__
DP::State::State(OP::Problem const* problem, Memory::MallocType mallocType) :
    cost(0),
    selectedValues(problem->variables.getCapacity(), mallocType),
    admissibleValues(problem->variables.getCapacity(), mallocType)
{}

__host__ __device__
bool DP::State::isAdmissible(OP::ValueType value) const
{
    return thrust::find(thrust::seq, admissibleValues.begin(), admissibleValues.end(), value) != admissibleValues.end();
}

__host__ __device__
std::byte* DP::State::mallocStorages(const OP::Problem* problem, unsigned int statesCount, Memory::MallocType mallocType)
{
    return Memory::safeMalloc(sizeOfStorage(problem) * statesCount, mallocType);
}

__host__ __device__
DP::State& DP::State::operator=(DP::State const& other)
{
    cost = other.cost;
    selectedValues.resize(other.selectedValues.getSize());
    thrust::copy(thrust::seq, other.selectedValues.begin(), other.selectedValues.end(), selectedValues.begin());
    admissibleValues.resize(other.admissibleValues.getSize());
    thrust::copy(thrust::seq, other.admissibleValues.begin(), other.admissibleValues.end(), admissibleValues.begin());
    return *this;
}

__host__ __device__
void DP::State::removeFromAdmissibles(OP::ValueType value)
{
    OP::ValueType const * const admissibleValuesEnd = thrust::remove(thrust::seq, admissibleValues.begin(), admissibleValues.end(), value);
    if (admissibleValuesEnd != admissibleValues.end())
    {
        unsigned int const size = admissibleValues.indexOf(admissibleValuesEnd);
        admissibleValues.resize(size);
    }
}

__host__ __device__
void DP::State::setInvalid()
{
    cost = DP::MaxCost;
    selectedValues.clear();
    admissibleValues.clear();
}

__host__ __device__
unsigned int DP::State::sizeOfStorage(OP::Problem const * problem)
{
    return
        Vector<OP::ValueType>::sizeOfStorage(problem->variables.getCapacity()) + // selectedValues
        Vector<OP::ValueType>::sizeOfStorage(problem->variables.getCapacity());  // admissibleValues
}

__host__ __device__
void DP::State::swap(DP::State& s0, DP::State& s1)
{
    thrust::swap(s0.cost, s1.cost);
    Vector<OP::ValueType>::swap(s0.selectedValues, s1.selectedValues);
    Vector<OP::ValueType>::swap(s0.admissibleValues, s1.admissibleValues);
}