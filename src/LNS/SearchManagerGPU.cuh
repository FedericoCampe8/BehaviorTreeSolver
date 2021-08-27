#pragma once
#include <thread>
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
    u64 iteration;
    SyncState<ProblemType, StateType> bestSolution;
    SyncState<ProblemType, StateType> neighborhoodSolution;
    private:
    ProblemType const * const problem;
    Options const * const options;
    OffloadBuffer<ProblemType,StateType> offloadBuffer;

    // Functions
    public:
    SearchManagerGPU(ProblemType const * problem, Options const * options, Memory::MallocType mallocType);
    void searchInitLoop(StatesPriorityQueue<StateType>* statesPriorityQueue, bool * timeout);
    void initializeRandomEngines(bool * timeout);
    void searchLnsLoop(StateType const * root, bool * timeout);
    private:
    void waitDevice() const;
    void doOffloadsAsync(LNS::SearchPhase searchPhase);
    void generateNeighbourhoodsAsync();
};


template<typename ProblemType, typename StateType>
SearchManagerGPU<ProblemType, StateType>::SearchManagerGPU(ProblemType const * problem, Options const * options, Memory::MallocType mallocType) :
    speed(0),
    done(false),
    iteration(0),
    problem(problem),
    options(options),
    bestSolution(problem, mallocType),
    neighborhoodSolution(problem, mallocType),
    offloadBuffer(problem, options->widthGpu, options->mddsGpu, options->probEq, options->probNeq, mallocType)
{
    bestSolution.state.invalidate();
}

template<typename ProblemType, typename StateType>
void SearchManagerGPU<ProblemType, StateType>::searchInitLoop(StatesPriorityQueue<StateType>* statesPriorityQueue, bool * timeout)
{
    done = false;
    iteration = 0;

    if(options->mddsGpu > 0)
    {
        while(not (statesPriorityQueue->isFull() or *timeout))
        {
            u64 const startTime = Chrono::now();

            // Initialize offload
            offloadBuffer.initializeOffload(statesPriorityQueue);

            if(not offloadBuffer.isEmpty())
            {
                // Offload
                doOffloadsAsync(LNS::SearchPhase::Init);
                waitDevice();

                //Finalize offload
                offloadBuffer.finalizeOffload(statesPriorityQueue);
                offloadBuffer.getBestSolution(LNS::SearchPhase::Init, &bestSolution);

                u64 const elapsedTime = max(Chrono::now() - startTime, 1ul);
                speed = offloadBuffer.getSize() * 1000 / elapsedTime;

                iteration += 1;
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
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
    u32 const blockSize = min(options->widthGpu * problem->maxBranchingFactor, 1024u);
    u32 const blockCount = searchPhase == LNS::SearchPhase::Init ? offloadBuffer.getSize() : offloadBuffer.getCapacity();
    u32 const sharedMemSize = DD::MDD<ProblemType, StateType>::sizeOfScratchpadMemory(problem, options->widthGpu);
    doOffloadKernel<<<blockCount, blockSize, sharedMemSize>>>(&offloadBuffer, searchPhase);
}

template<typename ProblemType, typename StateType>
__global__
void initializeRandomEnginesKernel(OffloadBuffer<ProblemType,StateType>* offloadBuffer, u32 randomSeed)
{
    if(threadIdx.x == 0)
    {
        u32 const index = blockIdx.x;
        offloadBuffer->initializeRandomEngines(randomSeed, index);
    }
}


template<typename ProblemType, typename StateType>
void SearchManagerGPU<ProblemType, StateType>::initializeRandomEngines(bool * timeout)
{
    // Random Numbers Generators
    if((not *timeout) and options->mddsGpu > 0)
    {
        u32 const blockSize = 1;
        u32 const blockCount = offloadBuffer.getCapacity();
        initializeRandomEnginesKernel<<<blockCount, blockSize>>>(&offloadBuffer, options->randomSeed);
        waitDevice();
    }
}

template<typename ProblemType, typename StateType>
void SearchManagerGPU<ProblemType, StateType>::searchLnsLoop(StateType const * root, bool * timeout)
{
    iteration = 0;

    if(options->mddsGpu > 0)
    {
        offloadBuffer.initializeOffload(root);

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

            u64 const elapsedTime = max(Chrono::now() - startTime, 1ul);
            speed = offloadBuffer.getCapacity() * 1000 / elapsedTime;

            iteration += 1;
        }
    }
}


template<typename ProblemType,typename StateType>
__global__
void generateNeighbourhoodKernel(OffloadBuffer<ProblemType,StateType>* offloadBuffer, StateType * solution)
{
    if(threadIdx.x == 0)
    {
        u32 const index = blockIdx.x;
        offloadBuffer->generateNeighborhood(&solution->selectedValues, index);
    }
}

template<typename ProblemType, typename StateType>
void SearchManagerGPU<ProblemType, StateType>::generateNeighbourhoodsAsync()
{
    u32 const blockSize = 1;
    u32 const blockCount = options->mddsGpu;
    generateNeighbourhoodKernel<<<blockCount, blockSize>>>(&offloadBuffer, &neighborhoodSolution.state);
}


template<typename ProblemType, typename StateType>
void SearchManagerGPU<ProblemType, StateType>::waitDevice() const
{
    cudaDeviceSynchronize();
}