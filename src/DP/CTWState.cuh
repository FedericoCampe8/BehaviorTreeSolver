#pragma once

#include "State.cuh"
#include "../OP/CTWProblem.cuh"

/*
namespace DP
{
    class CTWState : public State
    {
        // Members
        public:
        unsigned int s,m,n;
        int oldestOpenPairIdx;
        unsigned int openPairsCount;
        Array<unsigned int> blockingConstraintsCount;

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
    s(0),m(0),n(0),
    oldestOpenPairIdx(-1),
    openPairsCount(0),
    blockingConstraintsCount(problem->k, Memory::align<std::byte, unsigned int>(this->admissibleValues.endOfStorage()))
{}

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
    s = other.s;
    m = other.m;
    n = other.n;
    oldestOpenPairIdx = other.oldestOpenPairIdx;
    openPairsCount = other.openPairsCount;
    blockingConstraintsCount = other.blockingConstraintsCount;
    return *this;
}

__host__ __device__
unsigned int DP::CTWState::sizeOfStorage(OP::CTWProblem const * problem)
{
    return
        State::sizeOfStorage(problem) +
        Vector<unsigned  int>::sizeOfStorage(problem->k) + // blockingConstraintsCount
        Memory::AlignmentPadding; // alignment
}

__host__ __device__
void DP::CTWState::swap(DP::CTWState& ctws0, DP::CTWState& ctws1)
{
    State::swap(ctws0, ctws1);
    thrust::swap(ctws0.s, ctws1.s);
    thrust::swap(ctws0.m, ctws1.m);
    thrust::swap(ctws0.n, ctws1.n);
    thrust::swap(ctws0.oldestOpenPairIdx, ctws1.oldestOpenPairIdx);
    thrust::swap(ctws0.openPairsCount, ctws1.openPairsCount);
    Array<unsigned int>::swap(ctws0.blockingConstraintsCount, ctws1.blockingConstraintsCount);
}
*/