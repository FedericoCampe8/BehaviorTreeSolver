#pragma once

#include "State.cuh"
#include "../OP/SOPProblem.cuh"


namespace DP
{
    class SOPState : public State
    {
        // Members
        public:
        Array<u16> precedencesCount;

        // Functions
        public:
        __host__ __device__ SOPState(OP::SOPProblem const * problem, std::byte* storage);
        __host__ __device__ SOPState(OP::SOPProblem const * problem, Memory::MallocType mallocType);
        __host__ __device__ inline std::byte* endOfStorage() const;
        __host__ __device__ static std::byte* mallocStorages(OP::SOPProblem const *  problem, unsigned int statesCount, Memory::MallocType mallocType);
        __host__ __device__ SOPState& operator=(SOPState const & other);
        __host__ __device__ void print(bool endLine = true) const;
        __host__ __device__ static unsigned int sizeOfStorage(OP::SOPProblem const * problem);
        __host__ __device__ static void swap(SOPState& sops0, SOPState& sops1);
    };
}

__host__ __device__
DP::SOPState::SOPState(OP::SOPProblem const * problem, std::byte* storage) :
    State(problem, storage),
    precedencesCount(problem->variables.getCapacity(), Memory::align<u16>(this->State::endOfStorage()))
{}

__host__ __device__
DP::SOPState::SOPState(OP::SOPProblem const* problem, Memory::MallocType mallocType) :
    SOPState(problem, mallocStorages(problem,1,mallocType))
{}

__host__ __device__
std::byte* DP::SOPState::endOfStorage() const
{
    return precedencesCount.endOfStorage();
}

__host__ __device__
std::byte* DP::SOPState::mallocStorages(OP::SOPProblem const* problem, unsigned int statesCount, Memory::MallocType mallocType)
{
    return Memory::safeMalloc(sizeOfStorage(problem) * statesCount, mallocType);
}

__host__ __device__
DP::SOPState& DP::SOPState::operator=(DP::SOPState const & other)
{
    State::operator=(other);
    precedencesCount = other.precedencesCount;
    return *this;
}

__host__ __device__
void DP::SOPState::print(bool endLine) const
{
    State::print(endLine);
}


__host__ __device__
unsigned int DP::SOPState::sizeOfStorage(OP::SOPProblem const * problem)
{
    return
        State::sizeOfStorage(problem) +
        Array<u16>::sizeOfStorage(problem->variables.getCapacity()) + // precedencesCount
        Memory::DefaultAlignmentPadding * 2;
}

__host__ __device__
void DP::SOPState::swap(DP::SOPState& sops0, DP::SOPState& sops1)
{
    State::swap(sops0, sops1);
    Array<u16>::swap(sops0.precedencesCount, sops1.precedencesCount);
}
