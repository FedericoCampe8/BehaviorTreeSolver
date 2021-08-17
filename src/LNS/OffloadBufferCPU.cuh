#pragma once

#include <thread>
#include <random>
#include <LNS/OffloadBuffer.cuh>
#include <Options.h>
#include <thrust/extrema.h>

template<typename ProblemType, typename StateType>
class OffloadBufferCPU : public OffloadBuffer<ProblemType,StateType>
{
    // Members
    private:
    Array<std::thread> threads;
    Array<std::mt19937> rngs;

    // Functions
    public:
    OffloadBufferCPU(ProblemType const * problem, Options const & options);
    void initializeRngsAsync(u32 randomSeed);
    void generateNeighbourhoodsAsync(StateType const * solution);
    void wait();
    void doOffloadAsync(LNS::SearchPhase searchPhase);
    private:
    void initializeRngs(u32 randomSeed, u32 begin, u32 step, u32 end);
    void generateNeighbourhoods(StateType const * solution, u32 begin, u32 step, u32 end);
    void generateNeighbourhood(StateType const * solution, u32 index);
    void doOffload(LNS::SearchPhase searchPhase, u32 begin, u32 step, u32 end);

};

template<typename ProblemType, typename StateType>
OffloadBufferCPU<ProblemType, StateType>::OffloadBufferCPU(ProblemType const * problem, Options const & options) :
OffloadBuffer<ProblemType, StateType>(problem, options.widthCpu, options.parallelismCpu, options.eqProbability, options.neqProbability, Memory::MallocType::Std),
    threads(std::thread::hardware_concurrency(), Memory::MallocType::Std),
    rngs(this->capacity, Memory::MallocType::Std)
{}

template<typename ProblemType, typename StateType>
void OffloadBufferCPU<ProblemType, StateType>::generateNeighbourhoodsAsync(StateType const * solution)
{
    if(not this->isEmpty())
    {
        u32 const elements = this->capacity;
        u32 const threadsCount = threads.getCapacity();
        for (u32 threadIdx = 0; threadIdx < threadsCount; threadIdx += 1)
        {
            new (threads[threadIdx]) std::thread(&OffloadBufferCPU<ProblemType, StateType>::generateNeighbourhoods, this, solution,  threadIdx, threadsCount, elements);
        }
    }
}

template<typename ProblemType, typename StateType>
void OffloadBufferCPU<ProblemType, StateType>::generateNeighbourhoods(StateType const * solution, u32 begin, u32 step, u32 end)
{
   for(u32 index = begin; index < end; index += step)
   {
       generateNeighbourhood(solution, index);
   }
}

template<typename ProblemType, typename StateType>
void OffloadBufferCPU<ProblemType, StateType>::generateNeighbourhood(StateType const * solution, u32 index)
{
    std::uniform_real_distribution<float> distribution(0.0,1.0);
    this->neighbourhoods[index]->reset();
    for(u32 variableIdx = 0; variableIdx < solution->selectedValues.getCapacity(); variableIdx += 1)
    {
        OP::ValueType const value = *solution->selectedValues[variableIdx];
        float const random = distribution(*rngs[index]);
        this->neighbourhoods[index]->constraintVariable(variableIdx, value, random);
    }
}

template<typename ProblemType, typename StateType>
void OffloadBufferCPU<ProblemType, StateType>::wait()
{
    if(not this->isEmpty())
    {
        for (std::thread* thread = threads.begin(); thread != threads.end(); thread += 1)
        {
            if(thread->joinable())
            {
                thread->join();
            }
        }
    }
}

template<typename ProblemType, typename StateType>
void OffloadBufferCPU<ProblemType, StateType>::initializeRngsAsync(u32 randomSeed)
{
    if(not this->isEmpty())
    {
        u32 const elements = this->capacity;
        u32 const threadsCount = threads.getCapacity();
        for (u32 threadIdx = 0; threadIdx < threadsCount; threadIdx += 1)
        {
            new (threads[threadIdx]) std::thread(&OffloadBufferCPU<ProblemType, StateType>::initializeRngs, this, randomSeed, threadIdx, threadsCount, elements);
        }
    }
}

template<typename ProblemType, typename StateType>
void OffloadBufferCPU<ProblemType, StateType>::initializeRngs(u32 randomSeed, u32 begin, u32 step, u32 end)
{
    for (u32 rngIdx = begin; rngIdx < end; rngIdx += step)
    {
        new (rngs[rngIdx]) std::mt19937(randomSeed + rngIdx);
    }
}

template<typename ProblemType, typename StateType>
void OffloadBufferCPU<ProblemType, StateType>::doOffloadAsync(LNS::SearchPhase searchPhase)
{
    if(not this->isEmpty())
    {
        u32 const elements = searchPhase == LNS::SearchPhase::Init ? this->size : this->capacity;
        u32 const threadsCount = threads.getCapacity();
        for (u32 threadIdx = 0; threadIdx < threadsCount; threadIdx += 1)
        {
            new (threads[threadIdx]) std::thread(&OffloadBufferCPU<ProblemType, StateType>::doOffload, this, searchPhase, threadIdx, threadsCount, elements);
        }
    }
}


template<typename ProblemType, typename StateType>
void OffloadBufferCPU<ProblemType, StateType>::doOffload(LNS::SearchPhase searchPhase, u32 begin, u32 step, u32 end)
{
    for(u32 index = begin; index < end; index += step)
    {
        if(searchPhase == LNS::SearchPhase::Init)
        {
            this->mdds[index]->buildTopDown(this->neighbourhoods[index], this->topStates[index], this->bottomStates[index], false);
        }
        else
        {
            this->mdds[index]->buildTopDown(this->neighbourhoods[index], this->topStates[0], this->bottomStates[index], true);
        }
    }
}