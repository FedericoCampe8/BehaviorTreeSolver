#pragma once

#include "State.cuh"
#include "../OP/CTWProblem.cuh"


namespace DP
{
    class CTWState : public State
    {
        // Members
        public:
        u8 s,m,l,n;
        Vector<Pair<OP::ValueType>> interruptedPairs;
        Array<u8> precedencesCount;

        // Functions
        public:
        __host__ __device__ CTWState(OP::CTWProblem const * problem, std::byte* storage);
        __host__ __device__ CTWState(OP::CTWProblem const * problem, Memory::MallocType mallocType);
        __host__ __device__ inline std::byte* endOfStorage() const;
        __host__ __device__ static std::byte* mallocStorages(OP::CTWProblem const *  problem, unsigned int statesCount, Memory::MallocType mallocType);
        __host__ __device__ CTWState& operator=(CTWState const & other);
        __host__ __device__ void print(bool endLine = true) const;
        __host__ __device__ static unsigned int sizeOfStorage(OP::CTWProblem const * problem);
        __host__ __device__ static void swap(CTWState& ctws0, CTWState& ctws1);
    };
}

__host__ __device__
DP::CTWState::CTWState(OP::CTWProblem const * problem, std::byte* storage) :
    State(problem, storage),
    s(0), m(0), l(0), n(0),
    interruptedPairs(problem->b, Memory::align<Pair<OP::ValueType>>(this->State::endOfStorage())),
    precedencesCount(problem->variables.getCapacity(), Memory::align<u8>(interruptedPairs.endOfStorage()))
{}

__host__ __device__
DP::CTWState::CTWState(OP::CTWProblem const* problem, Memory::MallocType mallocType) :
    CTWState(problem, mallocStorages(problem,1,mallocType))
{}

__host__ __device__
std::byte* DP::CTWState::endOfStorage() const
{
    return precedencesCount.endOfStorage();
}

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
    l = other.l;
    n = other.n;
    interruptedPairs = other.interruptedPairs;
    precedencesCount = other.precedencesCount;
    return *this;
}

__host__ __device__
void DP::CTWState::print(bool endLine) const
{
    State::print(false);
    printf(" | S: %d | M: %d | L: %d | N: %d",s,m,l,n);
    printf(" | Interrupted: ");
    for(u32 openPairIdx = 0; openPairIdx < interruptedPairs.getSize(); openPairIdx += 1)
    {
        interruptedPairs.at(openPairIdx)->print(false);
        printf(" ");
    }
    printf(" | Admissibles: ");
    admissibleValuesMap.print(false);
    printf(" | Precedence: ");
    precedencesCount.print(false);
    printf(endLine ? "\n" : "");
}


__host__ __device__
unsigned int DP::CTWState::sizeOfStorage(OP::CTWProblem const * problem)
{
    return
        State::sizeOfStorage(problem) +
        Vector<Pair<OP::ValueType>>::sizeOfStorage(problem->b) + // interruptedPairs
        Array<u8>::sizeOfStorage(problem->variables.getCapacity()) + // precedencesCount
        Memory::DefaultAlignmentPadding;
}

__host__ __device__
void DP::CTWState::swap(DP::CTWState& ctws0, DP::CTWState& ctws1)
{
    State::swap(ctws0, ctws1);
    thrust::swap(ctws0.s, ctws1.s);
    thrust::swap(ctws0.m, ctws1.m);
    thrust::swap(ctws0.l, ctws1.l);
    thrust::swap(ctws0.n, ctws1.n);
    Vector<Pair<OP::ValueType>>::swap(ctws0.interruptedPairs, ctws1.interruptedPairs);
    Array<u8>::swap(ctws0.precedencesCount, ctws1.precedencesCount);
}