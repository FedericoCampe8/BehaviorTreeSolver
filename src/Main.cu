#include <cstdio>
#include <cinttypes>
#include <cstddef>
#include <algorithm>
#include <fstream>
#include <Containers/Buffer.cuh>
#include <Utils/Chrono.cuh>

#include <thread>

#include "BB/OffloadBuffer.cuh"
#include "DP/VRPModel.cuh"
#include "BB/PriorityQueue.cuh"

using namespace std;
using namespace Memory;
using namespace Chrono;
using namespace BB;
using namespace DD;
using namespace DP;
using namespace OP;
using namespace LNS;
using ModelType = VRPModel;
using ProblemType = VRProblem;
using StateType = VRPState;

// Auxiliary functions
void configGPU();

// Comparators
template<typename StateType>
bool hasSmallerCost(StateMetadata<StateType> const & sm0, StateMetadata<StateType> const & sm1);

// Queues
template<typename StateType>
bool boundsChk(DP::CostType bestCost, StateMetadata<StateType> const * stateMetadata);

template<typename StateType>
void updatePriorityQueue(DP::CostType bestCost, PriorityQueue<StateType>* priorityQueue, OffloadBuffer<StateType>* offloadBuffer);

// Search
template<typename StateType>
bool checkForBetterSolutions(StateType* bestSolution, StateType* currentSolution, OffloadBuffer<StateType>* offloadBuffer);

// Offload
template<typename StateType>
void prepareOffload(DP::CostType bestCost, PriorityQueue<StateType>* priorityQueue, OffloadBuffer<StateType>* offloadBuffer);

template<typename StateType>
void prepareOffload(StateMetadata<StateType> const * rootMetadata, OffloadBuffer<StateType>* offloadBuffer);

template<typename ModelType, typename ProblemType, typename StateType>
void doOffloadCpuAsync(DD::MDD<ModelType, ProblemType, StateType> const * mdd, OffloadBuffer<StateType>* cpuOffloadBuffer, Vector<std::thread>* cpuThreads, std::byte* scratchpadMem);

template<typename ModelType, typename ProblemType, typename StateType>
void doOffloadGpuAsync(DD::MDD<ModelType, ProblemType, StateType> const * mdd, OffloadBuffer<StateType>* gpuOffloadBuffer);

template<typename ModelType, typename ProblemType, typename StateType>
__host__ __device__ void doOffload(DD::MDD<ModelType,ProblemType,StateType> const * mdd, OffloadBuffer<StateType>* offloadBuffer, unsigned int index, std::byte* scratchpadMem);

template<typename ModelType, typename ProblemType, typename StateType>
__global__ void doOffloadKernel(DD::MDD<ModelType,ProblemType,StateType> const * mdd, OffloadBuffer<StateType>* offloadBuffer);

void waitOffloadCpu(Vector<std::thread>* cpuThreads);

void waitOffloadGpu();

// Debug
void printElapsedTime(uint64_t elapsedTimeMs);

void clearLine();

int main(int argc, char ** argv)
{
    // Input parsing
    char const * problemFileName = argv[1];
    unsigned int const queueMaxSize = std::stoi(argv[2]);
    unsigned int const timeoutSeconds = std::stoi(argv[3]);
    unsigned int const cpuMaxWidth = std::stoi(argv[4]);
    unsigned int const cpuMaxParallelism = std::stoi(argv[5]);
    unsigned int const gpuMaxWidth = std::stoi(argv[6]);
    unsigned int const gpuMaxParallelism = std::stoi(argv[7]);
    unsigned int const lnsEqPercentage = std::stoi(argv[8]);
    unsigned int const lnsNeqPercentage = std::stoi(argv[9]);
    unsigned int const randomSeed = std::stoi(argv[10]);
    assert(lnsEqPercentage + lnsNeqPercentage <= 100);

    // *******************
    // Data initialization
    // *******************

    // Context initialization
    std::mt19937 rng(randomSeed);
    MallocType gpuMallocType = MallocType::Std;
    if (gpuMaxParallelism > 0)
    {
        gpuMallocType = MallocType::Managed;
        configGPU();
    };

    // Problems
    ProblemType* const cpuProblem = VRProblem::parseGrubHubInstance(problemFileName, MallocType::Std);
    ProblemType* const gpuProblem = VRProblem::parseGrubHubInstance(problemFileName, gpuMallocType);

    // Models
    unsigned int memorySize = sizeof(ModelType);
    std::byte* memory = safeMalloc(memorySize, MallocType::Std);
    ModelType * const cpuModel = new (memory) ModelType(cpuProblem);
    memory = safeMalloc(memorySize, gpuMallocType);
    ModelType * const gpuModel = new (memory) ModelType(gpuProblem);

    // MDDs
    memorySize = sizeof(MDD<ModelType,ProblemType,StateType>);
    memory = safeMalloc(memorySize, MallocType::Std);
    MDD<ModelType,ProblemType,StateType>* const cpuMdd = new (memory) MDD<ModelType,ProblemType,StateType>(cpuModel, cpuMaxWidth);
    memory = safeMalloc(memorySize, gpuMallocType);
    MDD<ModelType,ProblemType,StateType>* const gpuMdd = new (memory) MDD<ModelType,ProblemType,StateType>(gpuModel, gpuMaxWidth);

    // Context initialization
    Vector<std::thread>* cpuThreads = new Vector<std::thread>(cpuMaxParallelism, MallocType::Std);
    std::byte* scratchpadMem = safeMalloc(cpuMdd->scratchpadMemSize * cpuMaxParallelism, MallocType::Std);

    // Queues
    PriorityQueue<StateType> priorityQueue(cpuProblem, hasSmallerCost<StateType>, queueMaxSize);

    // Offload
    memorySize = sizeof(OffloadBuffer<StateType>);
    memory = safeMalloc(memorySize, MallocType::Std);
    OffloadBuffer<StateType>* cpuOffloadBuffer = new (memory) OffloadBuffer<StateType>(cpuMdd, cpuMaxParallelism, MallocType::Std);
    memory = safeMalloc(memorySize, gpuMallocType);
    OffloadBuffer<StateType>* gpuOffloadBuffer = new (memory) OffloadBuffer<StateType>(gpuMdd, gpuMaxParallelism, gpuMallocType);

    // Solutions
    memorySize = sizeof(StateType);
    memory = safeMalloc(memorySize, MallocType::Std);
    StateType* bestSolution = new (memory) StateType(cpuProblem, StateType::mallocStorages(cpuProblem, 1, MallocType::Std));
    memory = safeMalloc(memorySize, MallocType::Std);
    StateType* currentSolution = new (memory) StateType(cpuProblem, StateType::mallocStorages(cpuProblem, 1, MallocType::Std));

    // Root
    memorySize = sizeof(StateType);
    memory = safeMalloc(memorySize, MallocType::Std);
    StateType* root = new (memory) StateType(cpuProblem, StateType::mallocStorages(cpuProblem, 1, MallocType::Std));
    cpuModel->makeRoot(root);

    // Search
    unsigned int iterationsCount = 0;
    enum SearchStatus {BB, LNS} searchStatus = SearchStatus::BB;
    unsigned int visitedStatesCount = 0;

    // ************
    // Begin search
    // ************

    // Enqueue root
    StateMetadata<StateType> const rootMetadata(0, DP::MaxCost, root);
    priorityQueue.insert(&rootMetadata);

    uint64_t searchStartTime = now();
    do
    {
        switch(searchStatus)
        {
            case SearchStatus::BB:
            {
                clearLine();
                printf("[INFO] Branch and bound");

                if (priorityQueue.isFull())
                {
                    searchStatus = SearchStatus::LNS;
                }

                prepareOffload<StateType>(currentSolution->cost, &priorityQueue, cpuOffloadBuffer);
                prepareOffload<StateType>(currentSolution->cost, &priorityQueue, gpuOffloadBuffer);
            }
                break;
            case SearchStatus::LNS:
            {
                clearLine();
                printf("[INFO] Large neighborhood search");

                prepareOffload<StateType>(&rootMetadata, cpuOffloadBuffer);
                prepareOffload<StateType>(&rootMetadata, gpuOffloadBuffer);
                cpuOffloadBuffer->generateNeighbourhoods(currentSolution, lnsEqPercentage, lnsNeqPercentage, &rng);
                gpuOffloadBuffer->generateNeighbourhoods(currentSolution, lnsEqPercentage, lnsNeqPercentage, &rng);
                currentSolution->reset();
            }
                break;
        }

        uint64_t cpuOffloadStartTime = now();
        doOffloadCpuAsync(cpuMdd, cpuOffloadBuffer, cpuThreads, scratchpadMem);

        uint64_t gpuOffloadStartTime = now();
        doOffloadGpuAsync(gpuMdd, gpuOffloadBuffer);

        waitOffloadCpu(cpuThreads);
        waitOffloadGpu();

        visitedStatesCount += cpuOffloadBuffer->getSize();
        visitedStatesCount += gpuOffloadBuffer->getSize();

        bool foundBetterSolution =
                checkForBetterSolutions(bestSolution, currentSolution, cpuOffloadBuffer) or
                checkForBetterSolutions(bestSolution, currentSolution, gpuOffloadBuffer);

        updatePriorityQueue(currentSolution->cost, &priorityQueue, cpuOffloadBuffer);
        updatePriorityQueue(currentSolution->cost, &priorityQueue, gpuOffloadBuffer);

        if(foundBetterSolution)
        {
            clearLine();
            printf("[INFO] Better solution found: ");
            bestSolution->selectedValues.print(false);
            printf(" | Value: %u", bestSolution->cost);
            printf(" | Time: ");
            printElapsedTime(now() - searchStartTime);
            printf(" | Iterations: %u", iterationsCount);
            printf(" | Visited states: %u\n", visitedStatesCount);
        }
        else
        {
            unsigned long int cpuSpeed = 0;
            if (cpuOffloadBuffer->getSize() > 0)
            {
                uint64_t cpuOffloadElapsedTime = max(1ul, now() - cpuOffloadStartTime);
                cpuSpeed = cpuOffloadBuffer->getSize() * 1000 / cpuOffloadElapsedTime;
            }

            unsigned long int gpuSpeed = 0;
            if (gpuOffloadBuffer->getSize() > 0)
            {
                uint64_t gpuOffloadElapsedTime = max(1ul, now() - gpuOffloadStartTime);
                gpuSpeed = gpuOffloadBuffer->getSize() * 1000 / gpuOffloadElapsedTime;
            }
            printf(" | Solution: ");
            currentSolution->selectedValues.print(false);
            printf(" | Value: %u", currentSolution->cost);
            printf(" | Time: ");
            printElapsedTime(now() - searchStartTime);
            printf(" | Iteration: %u", iterationsCount);
            printf(" | Speed: %lu - %lu\r", cpuSpeed, gpuSpeed);
        }
        fflush(stdout);
        iterationsCount += 1;
    }
    while(now() - searchStartTime < timeoutSeconds * 1000);

    printf("[RESULT] Solution: ");
    bestSolution->selectedValues.print(false);
    printf(" | Value: %u", bestSolution->cost);
    printf(" | Time: ");
    printElapsedTime(now() - searchStartTime);
    printf(" | Iterations: %u", iterationsCount);
    printf(" | Visited states: %u\n", visitedStatesCount);

    return EXIT_SUCCESS;
}

void configGPU()
{
    //Heap
    std::size_t sizeHeap = 3ul * 1024ul * 1024ul * 1024ul;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, sizeHeap);

    //Stack
    size_t sizeStackThread = 4 * 1024;
    cudaDeviceSetLimit(cudaLimitStackSize, sizeStackThread);

    //Cache
    cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual );
}



template<typename StateType>
bool hasSmallerCost(StateMetadata<StateType> const & sm0, StateMetadata<StateType> const & sm1)
{
    unsigned int const cost0 = sm0.state->cost;
    unsigned int const cost1 = sm1.state->cost;
    return cost0 < cost1;
}

template<typename StateType>
bool boundsChk(DP::CostType bestCost, StateMetadata<StateType> const * stateMetadata)
{
    return
        stateMetadata->lowerbound < stateMetadata->upperbound and
        stateMetadata->lowerbound < bestCost and
        stateMetadata->state->cost < bestCost;
}


template<typename StateType>
void updatePriorityQueue(DP::CostType bestCost, PriorityQueue<StateType>* priorityQueue, OffloadBuffer<StateType>* offloadBuffer)
{
    for (unsigned int index = 0; index < offloadBuffer->getSize(); index += 1)
    {
        StateMetadata<StateType> const * const offloadedStateMetadatata = offloadBuffer->getStateMetadata(index);
        LightArray<StateType> cutset = offloadBuffer->getCutset(index);
        for (StateType* cutsetState = cutset.begin(); cutsetState != cutset.end(); cutsetState += 1)
        {
            StateMetadata<StateType> const cutsetStateMetadata(offloadedStateMetadatata->lowerbound, offloadedStateMetadatata->upperbound, cutsetState);
            if(boundsChk(bestCost, &cutsetStateMetadata))
            {
                if(not priorityQueue->isFull())
                {
                    priorityQueue->insert(&cutsetStateMetadata);
                }
            }
        };
    };
}


template<typename StateType>
bool checkForBetterSolutions(StateType* bestSolution, StateType* currentSolution, OffloadBuffer<StateType>* offloadBuffer)
{
    bool foundBetterSolution = false;

    for (unsigned int index = 0; index < offloadBuffer->getSize(); index += 1)
    {
        unsigned int upperbound = offloadBuffer->getStateMetadata(index)->upperbound;
        if (upperbound < currentSolution->cost)
        {
            *currentSolution = *offloadBuffer->getApproximateSolution(index);
        }

        if (upperbound < bestSolution->cost)
        {
            *bestSolution = *offloadBuffer->getApproximateSolution(index);
            foundBetterSolution = true;
        }
    };

    return foundBetterSolution;
}

template<typename StateType>
void prepareOffload(DP::CostType bestCost, PriorityQueue<StateType>* priorityQueue, OffloadBuffer<StateType>* offloadBuffer)
{
    offloadBuffer->clear();
    while (not (priorityQueue->isEmpty() or offloadBuffer->isFull()))
    {
        StateMetadata<StateType> const * const stateMetadata = priorityQueue->front();
        if (boundsChk(bestCost, stateMetadata))
        {
            offloadBuffer->enqueue(stateMetadata);
        }
        priorityQueue->erase(stateMetadata);
    }
}
template<typename StateType>
void prepareOffload(StateMetadata<StateType> const * rootMetadata, OffloadBuffer<StateType>* offloadBuffer)
{
    offloadBuffer->clear();
    while (not offloadBuffer->isFull())
    {
        offloadBuffer->enqueue(rootMetadata);
    }
}


template<typename ModelType, typename ProblemType, typename StateType>
void doOffloadCpuAsync(MDD<ModelType, ProblemType, StateType> const * mdd, OffloadBuffer<StateType>* cpuOffloadBuffer, Vector<std::thread>* cpuThreads, std::byte* scratchpadMem)
{
    if(not cpuOffloadBuffer->isEmpty())
    {
        cpuThreads->clear();
        for (unsigned int index = 0; index < cpuOffloadBuffer->getSize(); index += 1)
        {
            cpuThreads->resize(cpuThreads->getSize() + 1);
            new (cpuThreads->back()) std::thread(&doOffload<ModelType, ProblemType, StateType>, mdd, cpuOffloadBuffer, index, &scratchpadMem[mdd->scratchpadMemSize * index]);
        }
    }
}

template<typename ModelType, typename ProblemType, typename StateType>
void doOffloadGpuAsync(MDD<ModelType, ProblemType, StateType> const * mdd, OffloadBuffer<StateType>* gpuOffloadBuffer)
{
    if(not gpuOffloadBuffer->isEmpty())
    {
        doOffloadKernel<ModelType, ProblemType, StateType><<<gpuOffloadBuffer->getSize(), 1, mdd->scratchpadMemSize>>>(mdd, gpuOffloadBuffer);
    }
}

template<typename ModelType, typename ProblemType, typename StateType>
__host__ __device__
void doOffload(MDD<ModelType,ProblemType,StateType> const * mdd, OffloadBuffer<StateType>* offloadBuffer, unsigned int index, std::byte* scratchpadMem)
{
    //Preparation to build MDDs
    StateMetadata<StateType>* const stateMetadata = offloadBuffer->getStateMetadata(index);
    StateType const * const top = stateMetadata->state;
    LightVector<StateType> cutset = offloadBuffer->getCutset(index);
    StateType * const bottom = offloadBuffer->getApproximateSolution(index);
    LNS::Neighbourhood const * const neighbourhood = offloadBuffer->getNeighbourhood(index);

    //Build MDDs
    mdd->buildTopDown(MDD<ModelType,ProblemType,StateType>::Type::Relaxed, top, &cutset, bottom, neighbourhood, scratchpadMem);
    stateMetadata->lowerbound = bottom->cost;
    mdd->buildTopDown(MDD<ModelType,ProblemType,StateType>::Type::Restricted, top, &cutset, bottom, neighbourhood, scratchpadMem);
    stateMetadata->upperbound = bottom->cost;
}

template<typename ModelType, typename ProblemType, typename StateType>
__global__
void doOffloadKernel(DD::MDD<ModelType,ProblemType,StateType> const * mdd, OffloadBuffer<StateType>* offloadBuffer)
{
    extern __shared__ unsigned int sharedMem[];
    std::byte* scratchpadMem = reinterpret_cast<std::byte*>(sharedMem);

    if(threadIdx.x == 0)
    {
        doOffload(mdd, offloadBuffer, blockIdx.x, scratchpadMem);
    };
}

void waitOffloadCpu(Vector<std::thread>* cpuThreads)
{
    for (std::thread* thread = cpuThreads->begin(); thread != cpuThreads->end(); thread += 1)
    {
        if(thread->joinable())
        {
            thread->join();
        }
    }
}

void waitOffloadGpu()
{
    cudaDeviceSynchronize();
}


void printElapsedTime(uint64_t elapsedTimeMs)
{
    unsigned int ms = elapsedTimeMs;

    unsigned int h = ms / (1000 * 60 * 60);
    ms -= h * 1000 * 60 * 60;

    unsigned int m = ms / (1000 * 60);
    ms -= m * 1000 * 60;

    unsigned int s = ms / 1000;

    printf("%lums (%02uh%02um%02us)", elapsedTimeMs, h, m, s);
}

void clearLine()
{
    // ANSI clear line escape code
    printf("\33[2K\r");
}
