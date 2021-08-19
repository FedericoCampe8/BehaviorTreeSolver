#pragma once
#include <thread>
#include <random>
#include <curand_kernel.h>
#include <LNS/OffloadBuffer.cuh>
#include <Options.h>
#include <Utils/CUDA.cuh>
#include <Utils/Chrono.cuh>
#include <thrust/extrema.h>

template<typename ProblemType, typename StateType>
class SearchManagerGPU
{
    // Members
    public:
    u64 speed;
    bool done;
    SyncState<StateType> bestSolution;
    SyncState<StateType> neighborhoodSolution;
    private:
    ProblemType const * problem;
    Options const * options;
    OffloadBuffer<ProblemType,StateType> offloadBuffer;
    Array<curandStatePhilox4_32_10_t> rngs;

    // Functions
    public:
    SearchManagerGPU(ProblemType const * problem, Options const * options, Memory::MallocType mallocType);
    void searchInitLoop(StatesPriorityQueue<StateType>* statesPriorityQueue, bool * timeout);
    void initializeRngs(bool * timeout);
    void searchLnsLoop(bool * timeout);
    private:
    void waitDevice() const;
    void doOffloadsAsync(LNS::SearchPhase searchPhase);
    void generateNeighbourhoodsAsync();
};


template<typename ProblemType, typename StateType>
SearchManagerGPU<ProblemType, StateType>::SearchManagerGPU(ProblemType const * problem, Options const * options, Memory::MallocType mallocType) :
    speed(0),
    done(false),
    problem(problem),
    options(options),
    bestSolution(problem, mallocType),
    neighborhoodSolution(problem, mallocType),
    offloadBuffer(problem, options->widthGpu, options->mddsGpu, options->probEq, options->probNeq, mallocType),
    rngs(options->mddsGpu, mallocType)
{}

template<typename ProblemType, typename StateType>
void SearchManagerGPU<ProblemType, StateType>::searchInitLoop(StatesPriorityQueue<StateType>* statesPriorityQueue, bool * timeout)
{
    done = false;
    if(options->mddsGpu > 0)
    {
        while(not (statesPriorityQueue->isFull() or *timeout))
        {
            u64 const startTime = Chrono::now();

            // Initialize offload
            offloadBuffer.initializeOffload(statesPriorityQueue);

            // Offload
            doOffloadsAsync(LNS::SearchPhase::Init);
            waitDevice();

            //Finalize offload
            offloadBuffer.finalizeOffload(statesPriorityQueue);
            offloadBuffer.getBestSolution(LNS::SearchPhase::Init, &bestSolution);

            u64 const elapsedTime = Chrono::now() - startTime;
            speed = offloadBuffer.getSize() * 1000 / elapsedTime;
        }
    }
    done = true;
}

template<typename ProblemType, typename StateType>
__global__
void doOffloadKernel(OffloadBuffer<ProblemType,StateType>* offloadBuffer, LNS::SearchPhase searchPhase)
{
    u32 const index = blockIdx.x;
    offloadBuffer->doOffload(searchPhase, index);
}

template<typename ProblemType, typename StateType>
void SearchManagerGPU<ProblemType, StateType>::doOffloadsAsync(LNS::SearchPhase searchPhase)
{
    u32 const blockSize = Algorithms::nextPower2(options->widthGpu * problem->maxBranchingFactor);
    u32 const blockCount = searchPhase == LNS::SearchPhase::Init ? offloadBuffer.getSize() : offloadBuffer.getCapacity();
    u32 const sharedMemSize = DD::MDD<ProblemType, StateType>::sizeOfScratchpadMemory(problem, options->widthGpu);
    assert(blockSize <= 1024);
    doOffloadKernel<<<blockCount, blockSize, sharedMemSize>>>(&offloadBuffer, searchPhase);
}

__global__
void initializeRngKernel(Array<curandStatePhilox4_32_10_t>* rngs, u32 randomSeed)
{
    if(threadIdx.x == 0)
    {
        u32 const index = blockIdx.x;
        curand_init(randomSeed + index, 0, 0, rngs->at(index));
    }
}


template<typename ProblemType, typename StateType>
void SearchManagerGPU<ProblemType, StateType>::initializeRngs(bool * timeout)
{
    // Random Numbers Generators
    if((not *timeout) and options->mddsGpu > 0)
    {
        u32 const blockSize = 1;
        u32 const blockCount = rngs.getCapacity();
        assert(blockSize <= 1024);
        initializeRngKernel<<<blockCount, blockSize>>>(&rngs, options->randomSeed);
        waitDevice();
    }
}

template<typename ProblemType, typename StateType>
void SearchManagerGPU<ProblemType, StateType>::searchLnsLoop(bool * timeout)
{
    done = false;

    if(options->mddsCpu > 0)
    {
        while(not *timeout)
        {
            u64 const startTime = Chrono::now();

            // Generate neighborhoods
            neighborhoodSolution.mutex.lock();
            generateNeighbourhoodsAsync();
            waitDevice();
            neighborhoodSolution.mutex.unlock();

            // Offload
            doOffloadsAsync(LNS::SearchPhase::LNS);
            waitDevice();

            //Finalize offload
            offloadBuffer.getBestSolution(LNS::SearchPhase::LNS, &bestSolution);

            u64 const elapsedTime = Chrono::now() - startTime;
            speed = offloadBuffer.getSize() * 1000 / elapsedTime;
        }
    }

    done = true;
}


template<typename ProblemType,typename StateType>
__global__
void generateNeighbourhoodKernel(OffloadBuffer<ProblemType,StateType>* offloadBuffer, Array<curandStatePhilox4_32_10_t>* rngs, StateType * solution)
{
    if(threadIdx.x == 0)
    {
        u32 const index = blockIdx.x;
        offloadBuffer->generateNeighborhood(rngs, &solution->selectedValues, index);
    }
}

template<typename ProblemType, typename StateType>
void SearchManagerGPU<ProblemType, StateType>::generateNeighbourhoodsAsync()
{
    u32 const blockSize = 1;
    u32 const blockCount = options->mddsGpu;
    generateNeighbourhoodKernel<<<blockCount, blockSize>>>(&offloadBuffer, &rngs, &neighborhoodSolution.state);
}


template<typename ProblemType, typename StateType>
void SearchManagerGPU<ProblemType, StateType>::waitDevice() const
{
    cudaDeviceSynchronize();
}