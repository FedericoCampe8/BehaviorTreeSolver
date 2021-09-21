#pragma once

#include "State.cuh"
#include "../OP/OSSProblem.cuh"


namespace DP
{
    class OSSPState : public State
    {
        // Members
        public:
        Array<u16> tasks_start;
        Array<u16> jobs_makespan;
        Array<u16> machines_makespan;
        Array<u16> machines_progress;

        // Private
        LightArray<OP::ValueType> * const selectedValuesArray;


        // Functions
        public:
        __host__ __device__ OSSPState(OP::OSSProblem const * problem, std::byte* storage);
        __host__ __device__ OSSPState(OP::OSSProblem const * problem, Memory::MallocType mallocType);
        __host__ __device__ inline std::byte* endOfStorage() const;
        __host__ __device__ static std::byte* mallocStorages(OP::OSSProblem const *  problem, unsigned int statesCount, Memory::MallocType mallocType);
        __host__ __device__ OSSPState& operator=(OSSPState const & other);
        __host__ __device__ void print(bool endLine = true) const;
        __host__ __device__ static unsigned int sizeOfStorage(OP::OSSProblem const * problem);
        __host__ __device__ static void swap(OSSPState& ossps0, OSSPState& ossps1);
    };
}

__host__ __device__
DP::OSSPState::OSSPState(OP::OSSProblem const * problem, std::byte* storage) :
    State(problem, storage),
    selectedValuesArray(static_cast<LightArray<OP::ValueType>*>(&selectedValues)),
    tasks_start(problem->jobs * problem->machines, Memory::align<u16>(this->State::endOfStorage())),
    jobs_makespan(problem->jobs, Memory::align<u16>(tasks_start.endOfStorage())),
    machines_makespan(problem->machines, Memory::align<u16>(jobs_makespan.endOfStorage())),
    machines_progress(problem->machines, Memory::align<u16>(machines_makespan.endOfStorage()))
{
    selectedValues.resize(selectedValues.getCapacity());
}

__host__ __device__
DP::OSSPState::OSSPState(OP::OSSProblem const* problem, Memory::MallocType mallocType) :
    OSSPState(problem, mallocStorages(problem,1,mallocType))
{}

__host__ __device__
std::byte* DP::OSSPState::endOfStorage() const
{
    return machines_makespan.endOfStorage();
}

__host__ __device__
std::byte* DP::OSSPState::mallocStorages(OP::OSSProblem const* problem, unsigned int statesCount, Memory::MallocType mallocType)
{
    u64 size = static_cast<u64>(sizeOfStorage(problem)) * static_cast<u64>(statesCount);
    return Memory::safeMalloc(size, mallocType);
}

__host__ __device__
DP::OSSPState& DP::OSSPState::operator=(DP::OSSPState const & other)
{
    State::operator=(other);
    tasks_start = other.tasks_start;
    jobs_makespan = other.jobs_makespan;
    machines_makespan = other.machines_makespan;
    machines_progress = other.machines_progress;
    return *this;
}

__host__ __device__
void DP::OSSPState::print(bool endLine) const
{
    tasks_start.print(endLine);
}


__host__ __device__
unsigned int DP::OSSPState::sizeOfStorage(OP::OSSProblem const * problem)
{
    u32 size = 0;
    size += State::sizeOfStorage(problem);
    size += Array<u16>::sizeOfStorage(problem->jobs * problem->machines);// tasks_start
    size += Array<u16>::sizeOfStorage(problem->jobs); // jobs_makespan
    size += Array<u16>::sizeOfStorage(problem->machines); // machines_makespan
    size += Array<u16>::sizeOfStorage(problem->machines); // machines_progress
    size += Memory::DefaultAlignmentPadding * 5;
    return size;
}

__host__ __device__
void DP::OSSPState::swap(DP::OSSPState& ossps0, DP::OSSPState& ossps1)
{
    State::swap(ossps0, ossps1);
    Array<u16>::swap(ossps0.tasks_start, ossps1.tasks_start);
    Array<u16>::swap(ossps0.jobs_makespan, ossps1.jobs_makespan);
    Array<u16>::swap(ossps0.machines_makespan, ossps1.machines_makespan);
    Array<u16>::swap(ossps0.machines_progress, ossps1.machines_progress);
}