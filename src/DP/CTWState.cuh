#pragma once

#include "State.cuh"
#include "../OP/CTWProblem.cuh"

namespace DP
{
    class CTWState : public State
    {
        // Members
        public:
        Array<OP::ValueType> pairs;

        // Functions
        public:
        __host__ __device__ CTWState(OP::CTWProblem const * problem, std::byte* storage);
        __host__ __device__ CTWState(OP::CTWProblem const * problem, Memory::MallocType mallocType);
        __host__ __device__ static std::byte* mallocStorages(OP::CTWProblem const *  problem, unsigned int statesCount, Memory::MallocType mallocType);
        __host__ __device__ CTWState& operator=(CTWState const & other);
        __host__ __device__ static unsigned int sizeOfStorage(OP::CTWProblem const * problem);
        __host__ __device__ static void swap(CTWState& ctws0, CTWState& ctws1);
    };
}

__host__ __device__
DP::CTWState::CTWState(OP::CTWProblem const * problem, std::byte* storage) :
    State(problem, storage),
    pairs(problem->b * 2, Memory::align<std::byte,OP::ValueType>(this->admissibleValues.endOfStorage()))
{
    thrust::fill(thrust::seq,pairs.begin(),pairs.end(),OP::MaxValue);
}

__host__ __device__
DP::CTWState::CTWState(OP::CTWProblem const* problem, Memory::MallocType mallocType) :
    CTWState(problem, mallocStorages(problem,1,mallocType))
{}

__host__ __device__
std::byte* DP::CTWState::mallocStorages(OP::CTWProblem const* problem, unsigned int statesCount, Memory::MallocType mallocType)
{
    return Memory::safeMalloc(sizeOfStorage(problem) * statesCount, mallocType);
}

__host__ __device__
DP::CTWState& DP::CTWState::operator=(DP::CTWState const & other)
{
    State::operator=(other);
    thrust::copy(thrust::seq, other.pairs.begin(), other.pairs.end(), pairs.begin());
    return *this;
}

__host__ __device__
unsigned int DP::CTWState::sizeOfStorage(OP::CTWProblem const * problem)
{
    return
        State::sizeOfStorage(problem) +
        Array<OP::ValueType>::sizeOfStorage(problem->b * 2) + // pairsStatus
        Memory::AlignmentPadding; // alignment
}

__host__ __device__ void
DP::CTWState::swap(DP::CTWState& ctws0, DP::CTWState& ctws1)
{
    State::swap(ctws0, ctws1);
    Array<OP::ValueType>::swap(ctws0.pairs, ctws1.pairs);
}
