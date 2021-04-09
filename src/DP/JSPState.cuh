#pragma once

#include "State.cuh"
#include "../OP/JSProblem.cuh"


namespace DP
{
    class JSPState : public State
    {
        // Members
        public:
        Array<u16> tasks_start;
        Array<u16> jobs_progress;
        Array<u16> jobs_makespan;
        Array<u16> machines_makespan;


        // Functions
        public:
        __host__ __device__ JSPState(OP::JSProblem const * problem, std::byte* storage);
        __host__ __device__ JSPState(OP::JSProblem const * problem, Memory::MallocType mallocType);
        __host__ __device__ inline std::byte* endOfStorage() const;
        __host__ __device__ static std::byte* mallocStorages(OP::JSProblem const *  problem, unsigned int statesCount, Memory::MallocType mallocType);
        __host__ __device__ JSPState& operator=(JSPState const & other);
        __host__ __device__ void print(bool endLine = true) const;
        __host__ __device__ static unsigned int sizeOfStorage(OP::JSProblem const * problem);
        __host__ __device__ static void swap(JSPState& ctws0, JSPState& ctws1);
    };
}

__host__ __device__
DP::JSPState::JSPState(OP::JSProblem const * problem, std::byte* storage) :
    State(problem, storage),
    tasks_start(problem->jobs * problem->machines, Memory::align<u16>(this->State::endOfStorage())),
    jobs_progress(problem->jobs, Memory::align<u16>(tasks_start.endOfStorage())),
    jobs_makespan(problem->jobs, Memory::align<u16>(jobs_progress.endOfStorage())),
    machines_makespan(problem->machines, Memory::align<u16>(jobs_makespan.endOfStorage()))
{}

__host__ __device__
DP::JSPState::JSPState(OP::JSProblem const* problem, Memory::MallocType mallocType) :
    JSPState(problem, mallocStorages(problem,1,mallocType))
{}

__host__ __device__
std::byte* DP::JSPState::endOfStorage() const
{
    return machines_makespan.endOfStorage();
}

__host__ __device__
std::byte* DP::JSPState::mallocStorages(OP::JSProblem const* problem, unsigned int statesCount, Memory::MallocType mallocType)
{
    return Memory::safeMalloc(sizeOfStorage(problem) * statesCount, mallocType);
}

__host__ __device__
DP::JSPState& DP::JSPState::operator=(DP::JSPState const & other)
{
    State::operator=(other);
    tasks_start = other.tasks_start;
    jobs_progress = other.jobs_progress;
    jobs_makespan = other.jobs_makespan;
    machines_makespan = other.machines_makespan;
    return *this;
}

__host__ __device__
void DP::JSPState::print(bool endLine) const
{
    tasks_start.print(endLine);
}


__host__ __device__
unsigned int DP::JSPState::sizeOfStorage(OP::JSProblem const * problem)
{
    return
        State::sizeOfStorage(problem) +
        Array<u16>::sizeOfStorage(problem->jobs * problem->machines) + // tasks_start
        Array<u16>::sizeOfStorage(problem->jobs) + // jobs_progress
        Array<u16>::sizeOfStorage(problem->jobs) + // jobs_makespan
        Array<u16>::sizeOfStorage(problem->machines) + // machines_makespan
        Memory::DefaultAlignmentPadding * 5;
}

__host__ __device__
void DP::JSPState::swap(DP::JSPState& jsps0, DP::JSPState& jsps1)
{
    State::swap(jsps0, jsps1);
    Array<u16>::swap(jsps0.tasks_start, jsps1.tasks_start);
    Array<u16>::swap(jsps0.jobs_progress, jsps1.jobs_progress);
    Array<u16>::swap(jsps0.jobs_makespan, jsps1.jobs_makespan);
    Array<u16>::swap(jsps0.machines_makespan, jsps1.machines_makespan);
}