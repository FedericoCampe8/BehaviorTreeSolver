#include <algorithm>
#include <thread>
#include <curand_kernel.h>
#include <External/AnyOption/anyoption.h>
#include <Utils/Algorithms.cuh>
#include <Utils/Chrono.cuh>
#include "DP/TSPPDModel.cuh"
#include "DP/CTWPModel.cuh"
#include "DP/MOSPModel.cuh"
#include "DP/JSPModel.cuh"
#include "DP/SOPModel.cuh"
#include "OffloadBuffer.cuh"
#include "BB/PriorityQueue.cuh"
#include "Options.h"

using namespace std;
using namespace Memory;
using namespace Chrono;
using namespace BB;
using namespace DD;
using namespace DP;
using namespace OP;
using ProblemType = TSPPDProblem;
using StateType = TSPPDState;

// Auxiliary functions
AnyOption* parseOptions(int argc, char* argv[]);

void configGPU();

void initRNGs(Array<std::mt19937>* rngs, u32 seed);

void initRNGs(Array<curandState_t>* rngs, u32 seed);

__global__ void initRNGsKernel(Array<curandState_t>* rngs, u32 seed);

// Search
template<typename ProblemType, typename StateType>
void updatePriorityQueue(StateType* bestSolution, unsigned int * filteredStatesCount, PriorityQueue<StateType>* priorityQueue, OffloadBuffer<ProblemType, StateType>* offloadBuffer);

template<typename StateType>
bool boundsCheck(StateType* bestSolution, AugmentedState<StateType> const * augmentedState);

template<typename ProblemType, typename StateType>
bool checkForBetterSolutions(StateType* bestSolution, StateType* currentSolution, OffloadBuffer<ProblemType,StateType>* offloadBuffer);

// Offload
template<typename ProblemType, typename StateType>
void prepareOffload(StateType* bestSolution, unsigned int* filteredStatesCount, PriorityQueue<StateType>* priorityQueue, OffloadBuffer<ProblemType, StateType>* offloadBuffer);

template<typename ProblemType, typename StateType>
void prepareOffload(AugmentedState<StateType> const * augmentedState, OffloadBuffer<ProblemType,StateType>* offloadBuffer);

template<typename ProblemType, typename StateType>
void doOffloadCpuAsync(OffloadBuffer<ProblemType,StateType>* cpuOffloadBuffer, Vector<std::thread>* cpuThreads, bool onlyRestricted);

template<typename ProblemType, typename StateType>
void doOffloadGpuAsync(OffloadBuffer<ProblemType,StateType>* gpuOffloadBuffer, bool onlyRestricted);

template<typename ProblemType, typename StateType>
void doOffloadsAsync(OffloadBuffer<ProblemType,StateType>* cpuOffloadBuffer, Vector<std::thread>* cpuThreads, OffloadBuffer<ProblemType,StateType>* gpuOffloadBuffer, bool onlyRestricted);

template<typename ProblemType, typename StateType>
void doOffloadLoop(OffloadBuffer<ProblemType,StateType>* offloadBuffer, unsigned int begin, unsigned int step, unsigned int end, bool onlyRestricted);

template<typename ProblemType, typename StateType>
__global__ void doOffloadKernel(OffloadBuffer<ProblemType,StateType>* offloadBuffer, bool onlyRestricted);

void waitOffloadCpu(Vector<std::thread>* cpuThreads, uint64_t* cpuOffloadEndTime);

void waitOffloadGpu(uint64_t* gpuOffloadEndTime, OffloadBuffer<ProblemType,StateType>* gpuOffloadBuffer);

void waitOffloads(Vector<std::thread>* cpuThreads, uint64_t* cpuOffloadEndTime, uint64_t* gpuOffloadEndTime, OffloadBuffer<ProblemType,StateType>* gpuOffloadBuffer);

// Debug
void printElapsedTime(uint64_t elapsedTimeMs);

void clearLine();

int main(int argc, char* argv[])
{
    // Input parsing
    Options options;
    if (not options.parseOptions(argc, argv))
    {
        return EXIT_FAILURE;
    }
    else
    {
        options.printOptions();
    }

    // *******************
    // Data initialization
    // *******************

    // Context initialization
    if (options.parallelismGpu > 0)
    {
        configGPU();
    };

    // CPU
    Vector<std::thread>* cpuThreads = new Vector<std::thread>(std::thread::hardware_concurrency(), MallocType::Std);
    Array<std::mt19937>* rngsCpu = new Array<std::mt19937>(options.parallelismCpu, MallocType::Std);
    initRNGs(rngsCpu, options.randomSeed);
    ProblemType* const cpuProblem = parseInstance<ProblemType>(options.inputFilename, MallocType::Std);

    // GPU
    Array<curandState_t>* rngsGPU = new Array<curandState_t>(options.parallelismGpu, gpuMallocType);
    initRNGs(rngsGPU, options.randomSeed);
    ProblemType* const gpuProblem = parseInstance<ProblemType>(options.inputFilename, gpuMallocType);

    // PriorityQueue
    PriorityQueue<StateType> priorityQueue(cpuProblem, options.queueSize);

    // Offload
    unsigned int memorySize = sizeof(OffloadBuffer<ProblemType,StateType>);
    byte* memory = safeMalloc(memorySize, MallocType::Std);
    OffloadBuffer<ProblemType,StateType>* cpuOffloadBuffer = new (memory) OffloadBuffer<ProblemType,StateType>(cpuProblem, options.widthCpu, options.parallelismCpu, MallocType::Std);
    memory = safeMalloc(memorySize, gpuMallocType);
    OffloadBuffer<ProblemType,StateType>* gpuOffloadBuffer = new (memory) OffloadBuffer<ProblemType,StateType>(gpuProblem, options.widthGpu, options.parallelismGpu, gpuMallocType);

    // Solutions
    memorySize = sizeof(StateType);
    memory = safeMalloc(memorySize, MallocType::Std);
    StateType* bestSolution = new (memory) StateType(cpuProblem, MallocType::Std);
    bestSolution->cost = DP::MaxCost;
    memory = safeMalloc(memorySize, MallocType::Std);
    StateType* currentSolution = new (memory) StateType(cpuProblem, MallocType::Std);
    currentSolution->cost = DP::MaxCost;

    // Root
    memorySize = sizeof(StateType);
    memory = safeMalloc(memorySize, MallocType::Std);
    StateType* root = new (memory) StateType(cpuProblem, MallocType::Std);
    makeRoot(cpuProblem, root);
    AugmentedState<StateType> const augmentedRoot(DP::MaxCost, 0, root);
    priorityQueue.insert(&augmentedRoot);

    // Search
    unsigned int iterationsCount = 0;
    enum SearchStatus {BB, LNS} searchStatus = SearchStatus::BB;
    unsigned int visitedStatesCount = 0;
    unsigned int filteredStatesCount = 0;

    // ************
    // Begin search
    // ************
    clearLine();
    uint64_t searchStartTime = now();
    printf("[INFO] Start branch and bound search");
    printf(" | Time: ");
    printElapsedTime(now() - searchStartTime);
    printf("\n");
    do
    {

        uint64_t iterationStartTime = now();

        uint64_t cpuOffloadEndTime;
        uint64_t gpuOffloadEndTime;

        switch(searchStatus)
        {
            case SearchStatus::BB:
            {
                if (priorityQueue.isFull())
                {
                    searchStatus = SearchStatus::LNS;
                    clearLine();
                    printf("[INFO] Switch to LNS");
                    printf(" | Time: ");
                    printElapsedTime(now() - searchStartTime);
                    printf("\n");
                }

                prepareOffload(bestSolution, &filteredStatesCount, &priorityQueue, cpuOffloadBuffer);
                prepareOffload(bestSolution, &filteredStatesCount, &priorityQueue, gpuOffloadBuffer);
            }
                break;
            case SearchStatus::LNS:
            {
                prepareOffload(&augmentedRoot, cpuOffloadBuffer);
                prepareOffload(&augmentedRoot, gpuOffloadBuffer);
                cpuOffloadBuffer->generateNeighbourhoods(bestSolution, options.eqProbability, options.neqProbability, rngsCpu, cpuThreads);
                waitOffloads(cpuThreads, &cpuOffloadEndTime, &gpuOffloadEndTime, gpuOffloadBuffer);
                currentSolution->makeInvalid();
            }
                break;
        }

        uint64_t cpuOffloadStartTime = now();
        uint64_t gpuOffloadStartTime = now();
        doOffloadsAsync(cpuOffloadBuffer, cpuThreads, gpuOffloadBuffer, searchStatus == SearchStatus::LNS);
        waitOffloads(cpuThreads, &cpuOffloadEndTime, &gpuOffloadEndTime, gpuOffloadBuffer);

        visitedStatesCount += cpuOffloadBuffer->getSize();
        visitedStatesCount += gpuOffloadBuffer->getSize();

        bool foundBetterSolutionCpu = checkForBetterSolutions(bestSolution, currentSolution, cpuOffloadBuffer);
        bool foundBetterSolutionGpu = checkForBetterSolutions(bestSolution, currentSolution, gpuOffloadBuffer);

        updatePriorityQueue(bestSolution, &filteredStatesCount, &priorityQueue, cpuOffloadBuffer);
        updatePriorityQueue(bestSolution, &filteredStatesCount, &priorityQueue, gpuOffloadBuffer);

        uint64_t iterationEndTime = now();

        if(foundBetterSolutionCpu or foundBetterSolutionGpu)
        {
            clearLine();
            printf("[SOLUTION] Source: %s", foundBetterSolutionCpu ? "CPU" : "GPU");
            printf(" | Time: ");
            printElapsedTime(now() - searchStartTime);
            printf(" | Cost: %u", bestSolution->cost);
            printf(" | Solution: ");
            bestSolution->print(true);

            //printf(" | Iterations: %u", iterationsCount);
            //printf(" | States: %u - %u - %u\n", visitedStatesCount, priorityQueue.getSize(), filteredStatesCount);
        }
        else
        {
            if(options.statistics)
            {
                unsigned long int cpuSpeed = 0;
                if (cpuOffloadBuffer->getSize() > 0)
                {
                    uint64_t cpuOffloadElapsedTime = Algorithms::max(1ul, cpuOffloadEndTime - cpuOffloadStartTime);
                    cpuSpeed = cpuOffloadBuffer->getSize() * 1000 / cpuOffloadElapsedTime;
                }

                unsigned long int gpuSpeed = 0;
                if (gpuOffloadBuffer->getSize() > 0)
                {
                    uint64_t gpuOffloadElapsedTime = Algorithms::max(1ul, gpuOffloadEndTime - gpuOffloadStartTime);
                    gpuSpeed = gpuOffloadBuffer->getSize() * 1000 / gpuOffloadElapsedTime;
                }
                //printf("[INFO] Solution: ");
                //currentSolution->selectedValues.print(false);
                //currentSolution->print(false);
                printf("[INFO] Current value: %u", currentSolution->cost);
                printf(" | Time: ");
                printElapsedTime(now() - searchStartTime);
                printf(" | Iteration: %u ", iterationsCount);
                printf("(");
                printElapsedTime(iterationEndTime - iterationStartTime);
                printf("s) ");
                //printf(" | States: %u - %u - %u", visitedStatesCount, priorityQueue.getSize(), filteredStatesCount);
                printf(" | CPU: %lu MDD/s | GPU: %lu MDD/s\r", cpuSpeed, gpuSpeed);
            }
        }
        fflush(stdout);
        iterationsCount += 1;
    }
    while(now() - searchStartTime < options.timeout * 1000 and (not priorityQueue.isEmpty()));

    //clearLine();
    //bestSolution->selectedValues.print(false);
    //bestSolution->print(false);
    //printf(" | Value: %u", bestSolution->cost);
    //printf(" | Time: ");
    //printElapsedTime(now() - searchStartTime);
    //printf(" | Iterations: %u", iterationsCount);
    //printf(" | States: %u - %u - %u\n", visitedStatesCount, priorityQueue.getSize(), filteredStatesCount);

    return EXIT_SUCCESS;
}

void configGPU()
{
    //Heap
    std::size_t const sizeHeap = 4ul * 1024ul * 1024ul * 1024ul;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, sizeHeap);

    //Stack
    size_t const sizeStackThread = 4 * 1024;
    cudaDeviceSetLimit(cudaLimitStackSize, sizeStackThread);

    //Cache
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
}

template<typename ProblemType, typename StateType>
void updatePriorityQueue(StateType* bestSolution, unsigned int * filteredStatesCount, PriorityQueue<StateType>* priorityQueue, OffloadBuffer<ProblemType, StateType>* offloadBuffer)
{
    for (unsigned int index = 0; index < offloadBuffer->getSize(); index += 1)
    {
        AugmentedState<StateType> const * parentAugmentedState = offloadBuffer->getAugmentedState(index);
        if(boundsCheck(bestSolution, parentAugmentedState))
        {
            Vector<StateType> const * const cutset = &offloadBuffer->getMDD(index)->cutset;
            for (StateType* cutsetState = cutset->begin(); cutsetState != cutset->end(); cutsetState += 1)
            {
                if (not priorityQueue->isFull())
                {
                    AugmentedState<StateType> const childAugmentedState(parentAugmentedState->upperbound, parentAugmentedState->lowerbound, cutsetState);
                    priorityQueue->insert(&childAugmentedState);
                }
            };
        }
        else
        {
            *filteredStatesCount += 1;
        }
    };
}

template<typename StateType>
bool boundsCheck(StateType* bestSolution, AugmentedState<StateType> const * augmentedState)
{
    return
       augmentedState->lowerbound < augmentedState->upperbound and
        augmentedState->lowerbound < bestSolution->cost and
        augmentedState->state->cost <= bestSolution->cost;
}

template<typename ProblemType, typename StateType>
bool checkForBetterSolutions(StateType* bestSolution, StateType* currentSolution, OffloadBuffer<ProblemType,StateType>* offloadBuffer)
{
    bool foundBetterSolution = false;
    std::uniform_int_distribution<u32> randomDistribution(0, 1);
    for (unsigned int index = 0; index < offloadBuffer->getSize(); index += 1)
    {
        StateType const * const approximateSolution = &offloadBuffer->getMDD(index)->bottom;

        if(approximateSolution->cost < currentSolution->cost)
        {
            *currentSolution = *approximateSolution;
        }

        if (approximateSolution->cost < bestSolution->cost)
        {
            *bestSolution = *approximateSolution;
            foundBetterSolution = true;
        }
    };

    return foundBetterSolution;
}

template<typename ProblemType, typename StateType>
void prepareOffload(StateType* bestSolution, unsigned int * filteredStatesCount, PriorityQueue<StateType>* priorityQueue, OffloadBuffer<ProblemType, StateType>* offloadBuffer)
{
    offloadBuffer->clear();
    while (not (priorityQueue->isEmpty() or offloadBuffer->isFull()))
    {
        AugmentedState<StateType> const * const augmentedState = priorityQueue->front();
        if(boundsCheck(bestSolution, augmentedState))
        {
            offloadBuffer->enqueue(augmentedState);
        }
        else
        {
            *filteredStatesCount += 1;
        }
        priorityQueue->popFront();
    }
}

template<typename ProblemType, typename StateType>
void prepareOffload(AugmentedState<StateType> const * augmentedState, OffloadBuffer<ProblemType,StateType>* offloadBuffer)
{
    offloadBuffer->clear();
    while (not offloadBuffer->isFull())
    {
        offloadBuffer->enqueue(augmentedState);
    }
}

template<typename ProblemType, typename StateType>
void doOffloadCpuAsync(OffloadBuffer<ProblemType,StateType>* cpuOffloadBuffer, Vector<std::thread>* cpuThreads, bool onlyRestricted)
{
    cpuThreads->clear();
    unsigned int step = cpuThreads->getCapacity();
    unsigned int end = cpuOffloadBuffer->getSize();
    for (unsigned int begin = 0; begin < cpuThreads->getCapacity(); begin += 1)
    {
        cpuThreads->resize(cpuThreads->getSize() + 1);
        new (cpuThreads->back()) std::thread(doOffloadLoop<ProblemType,StateType>, cpuOffloadBuffer, begin, step, end, onlyRestricted);
    }
}

template<typename ProblemType, typename StateType>
void doOffloadGpuAsync(OffloadBuffer<ProblemType,StateType>* gpuOffloadBuffer, bool onlyRestricted)
{
    if(not gpuOffloadBuffer->isEmpty())
    {
        DD::MDD<ProblemType,StateType> const * const mdd = gpuOffloadBuffer->getMDD(0);
        unsigned int const blocksCount = gpuOffloadBuffer->getSize();
        unsigned int const blockSize = Algorithms::min(mdd->width * mdd->problem->maxBranchingFactor, 1024u);
        doOffloadKernel<ProblemType, StateType><<<blocksCount, blockSize, mdd->sizeOfScratchpadMemory()>>>(gpuOffloadBuffer, onlyRestricted);
    }
}

template<typename ProblemType, typename StateType>
void doOffloadsAsync(OffloadBuffer<ProblemType,StateType>* cpuOffloadBuffer, Vector<std::thread>* cpuThreads, OffloadBuffer<ProblemType,StateType>* gpuOffloadBuffer, bool onlyRestricted)
{
    doOffloadCpuAsync(cpuOffloadBuffer, cpuThreads, onlyRestricted);
    doOffloadGpuAsync(gpuOffloadBuffer, onlyRestricted);
}

template<typename ProblemType, typename StateType>
void doOffloadLoop(OffloadBuffer<ProblemType, StateType>* offloadBuffer, unsigned int begin, unsigned int step, unsigned int end, bool onlyRestricted)
{
    for(unsigned int index = begin; index < end; index += step)
    {
        offloadBuffer->doOffload(index, onlyRestricted);
    }
}


template<typename ProblemType, typename StateType>
__global__
void doOffloadKernel(OffloadBuffer<ProblemType,StateType>* offloadBuffer, bool onlyRestricted)
{
    offloadBuffer->doOffload(blockIdx.x, onlyRestricted);
}

void waitOffloadCpu(Vector<std::thread>* cpuThreads, uint64_t* cpuOffloadEndTime)
{
    for (std::thread* thread = cpuThreads->begin(); thread != cpuThreads->end(); thread += 1)
    {
        if(thread->joinable())
        {
            thread->join();
        }
    }
    *cpuOffloadEndTime = now();
}

void waitOffloadGpu(uint64_t* gpuOffloadEndTime, OffloadBuffer<ProblemType,StateType>* gpuOffloadBuffer)
{
    cudaDeviceSynchronize();
    *gpuOffloadEndTime = now();
}

void waitOffloads(Vector<std::thread>* cpuThreads, uint64_t* cpuOffloadEndTime, uint64_t* gpuOffloadEndTime, OffloadBuffer<ProblemType,StateType>* gpuOffloadBuffer)
{
    std::thread waitCpu(waitOffloadCpu, cpuThreads, cpuOffloadEndTime);
    std::thread waitGpu(waitOffloadGpu, gpuOffloadEndTime, gpuOffloadBuffer);

    waitCpu.join();
    waitGpu.join();
}

void printElapsedTime(uint64_t elapsedTimeMs)
{
    unsigned int ms = elapsedTimeMs;

    unsigned int h = ms / (1000 * 60 * 60);
    ms -= h * 1000 * 60 * 60;

    unsigned int m = ms / (1000 * 60);
    ms -= m * 1000 * 60;

    //unsigned int s = ms / 1000;

    //printf("%lums (%02uh%02um%02us)", elapsedTimeMs, h, m, s);
    printf("%.3f", ms / 1000.0);
}

void clearLine()
{
    // ANSI clear line escape code
    printf("\33[2K\r");
}

void initRNGs(Array<std::mt19937>* rngs, u32 const seed)
{
    for(u32 i = 0; i < rngs->getCapacity(); i += 1)
    {
        new (rngs->at(i)) std::mt19937(seed + i);
    }
}

void initRNGs(Array<curandState_t>* rngs, u32 seed)
{
    initRNGsKernel<<<rngs->getCapacity(), 1>>>(rngs, seed);
    cudaDeviceSynchronize();
}

__global__ void initRNGsKernel(Array<curandState_t>* rngs, u32 const seed)
{
    u32 i = blockIdx.x;
    curand_init(seed + i, 0, 0, rngs->at(i));
}
