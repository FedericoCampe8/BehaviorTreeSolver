#pragma once

#include <thread>
#include <mutex>
#include <random>
#include <LNS/OffloadBuffer.cuh>
#include <LNS/SyncState.cuh>
#include <Options.h>
#include <thrust/extrema.h>
#include <Utils/Chrono.cuh>

template<typename ProblemType, typename StateType>
class SearchManagerCPU
{
    // Members
    public:
    u64 speed;
    bool done;
    u64 iteration;
    SyncState<ProblemType, StateType> bestSolution;
    SyncState<ProblemType, StateType> neighborhoodSolution;
    private:
    ProblemType const * const problem;
    Options const * const options;
    OffloadBuffer<ProblemType,StateType> offloadBuffer;
    Array<std::thread> threads;

    // Functions
    public:
    SearchManagerCPU(ProblemType const * problem, Options const * options);
    void searchInitLoop(StatesPriorityQueue<StateType>* statesPriorityQueue, bool * timeout);
    void initializeRandomEngines(bool * timeout);
    void searchLnsLoop(StateType const * root, bool * timeout);
    private:
    void waitThreads();
    void doOffloadsAsync(LNS::SearchPhase searchPhase);
    void doOffload(LNS::SearchPhase searchPhase, u32 beginIdx, u32 endIdx);
    void initializeRandomEnginesAsync(u32 beginIdx, u32 endIdx);
    void generateNeighbourhoodsAsync();
    void generateNeighbourhoods(u32 beginIdx, u32 endIdx);
};

template<typename ProblemType, typename StateType>
SearchManagerCPU<ProblemType, StateType>::SearchManagerCPU(ProblemType const * problem, Options const * options) :
    speed(0),
    done(false),
    iteration(0),
    problem(problem),
    options(options),
    bestSolution(problem, Memory::MallocType::Std),
    neighborhoodSolution(problem, Memory::MallocType::Std),
    offloadBuffer(problem, options->widthCpu, options->mddsCpu, options->probEq, options->probNeq, Memory::MallocType::Std),
    threads(std::thread::hardware_concurrency(), Memory::MallocType::Std)
{
    bestSolution.state.invalidate();
}

template<typename ProblemType, typename StateType>
void SearchManagerCPU<ProblemType, StateType>::searchInitLoop(StatesPriorityQueue<StateType>* statesPriorityQueue, bool * timeout)
{
    done = false;
    iteration = 0;

    if(options->mddsCpu > 0)
    {
        while(not (statesPriorityQueue->isFull() or statesPriorityQueue->isEmpty() or *timeout))
        {
            u64 const startTime = Chrono::now();

            // Initialize offload
            offloadBuffer.initializeOffload(statesPriorityQueue);

            if(not offloadBuffer.isEmpty())
            {
                // Offload
                doOffloadsAsync(LNS::SearchPhase::Init);
                waitThreads();

                //Finalize offload
                offloadBuffer.finalizeOffload(statesPriorityQueue);
                offloadBuffer.getBestSolution(LNS::SearchPhase::Init, &bestSolution);

                u64 const elapsedTime = max(Chrono::now() - startTime, 1ul);
                speed = offloadBuffer.getSize() * 1000 / elapsedTime;

                iteration += 1;
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
    }

    done = true;
}

template<typename ProblemType, typename StateType>
void SearchManagerCPU<ProblemType, StateType>::initializeRandomEngines(bool * timeout)
{
    // Random Numbers Generators
    if((not *timeout) and options->mddsCpu > 0)
    {
        u32 elements = offloadBuffer.getCapacity();
        u32 threadsCount = threads.getCapacity();
        for (u32 threadIdx = 0; threadIdx < threadsCount; threadIdx += 1)
        {
            Pair<u32> indicesBeginEnd = Algorithms::getBeginEndIndices(threadIdx,threadsCount, elements);
            new (threads[threadIdx]) std::thread(&SearchManagerCPU<ProblemType, StateType>::initializeRandomEnginesAsync, this, indicesBeginEnd.first, indicesBeginEnd.second);
        }
        waitThreads();
    }
}

template<typename ProblemType, typename StateType>
void SearchManagerCPU<ProblemType, StateType>::searchLnsLoop(StateType const * root, bool * timeout)
{
    done = false;
    iteration = 0;

    if(options->mddsCpu > 0)
    {
        offloadBuffer.initializeOffload(root);

        while(not *timeout)
        {
            u64 const startTime = Chrono::now();

            // Generate neighborhoods
            neighborhoodSolution.mutex.lock();
            generateNeighbourhoodsAsync();
            waitThreads();
            neighborhoodSolution.mutex.unlock();

            // Offload
            doOffloadsAsync(LNS::SearchPhase::LNS);
            waitThreads();

            //Finalize offload
            offloadBuffer.getBestSolution(LNS::SearchPhase::LNS, &bestSolution);

            u64 const elapsedTime = max(Chrono::now() - startTime, 1ul);
            speed = offloadBuffer.getCapacity() * 1000 / elapsedTime;

            iteration += 1;
        }
    }

    done = true;
}

template<typename ProblemType, typename StateType>
void SearchManagerCPU<ProblemType, StateType>::initializeRandomEnginesAsync(u32 beginIdx, u32 endIdx)
{
    for (u32 index = beginIdx; index < endIdx; index += 1)
    {
        offloadBuffer.initializeRandomEngines(options->randomSeed, index);
    }
}

template<typename ProblemType, typename StateType>
void SearchManagerCPU<ProblemType, StateType>::waitThreads()
{
    for (std::thread* thread = threads.begin(); thread != threads.end(); thread += 1)
    {
        if(thread->joinable())
        {
            thread->join();
        }
    }
}

template<typename ProblemType, typename StateType>
void SearchManagerCPU<ProblemType, StateType>::generateNeighbourhoodsAsync()
{
    u32 const elements = options->mddsCpu;
    u32 const threadsCount = threads.getCapacity();
    for (u32 threadIdx = 0; threadIdx < threadsCount; threadIdx += 1)
    {
        Pair<u32> indicesBeginEnd = Algorithms::getBeginEndIndices(threadIdx,threadsCount, elements);
        new (threads[threadIdx]) std::thread(&SearchManagerCPU<ProblemType, StateType>::generateNeighbourhoods, this, indicesBeginEnd.first, indicesBeginEnd.second);
    }
}

template<typename ProblemType, typename StateType>
void SearchManagerCPU<ProblemType, StateType>::generateNeighbourhoods(u32 beginIdx, u32 endIdx)
{
   for(u32 index = beginIdx; index < endIdx; index += 1)
   {
        offloadBuffer.generateNeighborhood(&neighborhoodSolution.state.selectedValues, index);
   }
}

template<typename ProblemType, typename StateType>
void SearchManagerCPU<ProblemType, StateType>::doOffload(LNS::SearchPhase searchPhase, u32 beginIdx, u32 endIdx)
{
    for(u32 index = beginIdx; index < endIdx; index += 1)
    {
        offloadBuffer.doOffload(searchPhase, index);
    }
}

template<typename ProblemType, typename StateType>
void SearchManagerCPU<ProblemType, StateType>::doOffloadsAsync(LNS::SearchPhase searchPhase)
{
    u32 const elements = offloadBuffer.getSize();
    u32 const threadsCount = threads.getCapacity();
    for (u32 threadIdx = 0; threadIdx < threadsCount; threadIdx += 1)
    {
        Pair<u32> indicesBeginEnd = Algorithms::getBeginEndIndices(threadIdx,threadsCount, elements);
        new (threads[threadIdx]) std::thread(&SearchManagerCPU<ProblemType, StateType>::doOffload, this, searchPhase, indicesBeginEnd.first, indicesBeginEnd.second);
    }
}

