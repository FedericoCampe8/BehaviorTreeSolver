#pragma once

#include <thread>
#include <Utils/Memory.cuh>
#include <OP/Problem.cuh>

template<typename StateType>
class SyncState
{
    public:
    StateType state;
    std::mutex mutex;

    SyncState(OP::Problem const * problem, std::byte* storage);
    SyncState(OP::Problem const * problem, Memory::MallocType mallocType);
};

template<typename StateType>
SyncState<StateType>::SyncState(OP::Problem const * problem, std::byte* storage) :
    state(problem, storage),
    mutex()
{}

template<typename StateType>
SyncState<StateType>::SyncState(OP::Problem const * problem, Memory::MallocType mallocType) :
state(problem, mallocType),
    mutex()
{}
