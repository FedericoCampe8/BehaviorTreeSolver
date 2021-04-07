#pragma once

#include "State.cuh"
#include "../OP/MOSProblem.cuh"


namespace DP
{
    class MOSPState : public State
    {
        // Members
        public:
        u16 clientsToClose;
        u16 openStacksCount;
        u16 maxOpenStacksCount;
        Array<u16> productsToDo;


        // Functions
        public:
        __host__ __device__ MOSPState(OP::MOSProblem const * problem, std::byte* storage);
        __host__ __device__ MOSPState(OP::MOSProblem const * problem, Memory::MallocType mallocType);
        __host__ __device__ inline std::byte* endOfStorage() const;
        __host__ __device__ static std::byte* mallocStorages(OP::MOSProblem const *  problem, unsigned int statesCount, Memory::MallocType mallocType);
        __host__ __device__ MOSPState& operator=(MOSPState const & other);
        __host__ __device__ void print(bool endLine = true) const;
        __host__ __device__ static unsigned int sizeOfStorage(OP::MOSProblem const * problem);
        __host__ __device__ static void swap(MOSPState& ctws0, MOSPState& ctws1);
    };
}

__host__ __device__
DP::MOSPState::MOSPState(OP::MOSProblem const * problem, std::byte* storage) :
    State(problem, storage),
    clientsToClose(0),
    openStacksCount(0),
    maxOpenStacksCount(0),
    productsToDo(problem->clients, Memory::align<u16>(this->State::endOfStorage()))
{}

__host__ __device__
DP::MOSPState::MOSPState(OP::MOSProblem const* problem, Memory::MallocType mallocType) :
    MOSPState(problem, mallocStorages(problem,1,mallocType))
{}

__host__ __device__
std::byte* DP::MOSPState::endOfStorage() const
{
    return productsToDo.endOfStorage();
}

__host__ __device__
std::byte* DP::MOSPState::mallocStorages(OP::MOSProblem const* problem, unsigned int statesCount, Memory::MallocType mallocType)
{
    return Memory::safeMalloc(sizeOfStorage(problem) * statesCount, mallocType);
}

__host__ __device__
DP::MOSPState& DP::MOSPState::operator=(DP::MOSPState const & other)
{
    State::operator=(other);
    clientsToClose = other.clientsToClose;
    openStacksCount = other.openStacksCount;
    maxOpenStacksCount = other.maxOpenStacksCount;
    productsToDo = other.productsToDo;
    return *this;
}

__host__ __device__
void DP::MOSPState::print(bool endLine) const
{
    State::print(false);
    printf(" | Admissibles: ");
    admissibleValuesMap.print(false);
    printf(endLine ? "\n" : "");
}


__host__ __device__
unsigned int DP::MOSPState::sizeOfStorage(OP::MOSProblem const * problem)
{
    return
        State::sizeOfStorage(problem) +
        Array<u16>::sizeOfStorage(problem->clients) + // productsToDo
        Memory::DefaultAlignmentPadding;
}

__host__ __device__
void DP::MOSPState::swap(DP::MOSPState& mosps0, DP::MOSPState& mosps1)
{
    State::swap(mosps0, mosps1);
    thrust::swap(mosps0.clientsToClose, mosps1.clientsToClose);
    thrust::swap(mosps0.openStacksCount, mosps1.openStacksCount);
    thrust::swap(mosps0.maxOpenStacksCount, mosps1.maxOpenStacksCount);
    Array<u16>::swap(mosps0.productsToDo, mosps1.productsToDo);
}