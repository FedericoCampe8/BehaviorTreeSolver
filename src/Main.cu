#include <cstdio>
#include <cinttypes>
#include <cstddef>
#include <new>
#include <utility>
#include <algorithm>
#include <fstream>
#include <Containers/Buffer.cuh>
#include <Utils/Memory.cuh>
#include <Utils/Chrono.cuh>
#include <External/NlohmannJson.hpp>
#include <thread>
#include <thrust/equal.h>

#include "BB/OffloadQueue.cuh"
#include "BB/PriorityQueuesManager.cuh"
#include "DD/MDD.cuh"
#include "DP/VRPModel.cuh"
#include "TS/Neighbourhood.cuh"

using namespace std;
using json = nlohmann::json;
using namespace Memory;
using namespace Chrono;
using namespace BB;
using namespace DD;
using namespace DP;
using namespace OP;
using namespace TS;
using ModelType = VRPModel;
using ProblemType = VRProblem;
using StateType = VRPState;

// Auxiliary functions
void configGPU();
OP::VRProblem* parseGrubHubInstance(char const * problemFileName, Memory::MallocType mallocType);

// Comparators
template<typename StateType>
bool hasSmallerCost(QueuedState<StateType> const & queuedState0, QueuedState<StateType> const & queuedState1);

// Comparators
template<typename StateType>
bool hasBiggerCost(QueuedState<StateType> const & queuedState0, QueuedState<StateType> const & queuedState1);

template<typename StateType>
bool hasRandomPriority(QueuedState<StateType> const & queuedState0, QueuedState<StateType> const & queuedState1);

template<typename StateType>
bool hasLessSelections(QueuedState<StateType> const & queuedState0, QueuedState<StateType> const & queuedState1);

// Queues
template<typename StateType>
bool boundsChk(unsigned int bestCost, StateMetadata<StateType> const * stateMetadata);

template<typename StateType>
void updatePriorityQueues(unsigned int bestCost, PriorityQueuesManager<StateType>* priorityQueuesManager, OffloadQueue<StateType>* offloadQueue);

// Search
template<typename StateType>
bool checkForBetterSolutions(StateType* bestSolution, StateType* currentSolution, OffloadQueue<StateType>* offloadQueue);

// Offload
template<typename StateType>
void prepareOffload(unsigned int bestCost, MaxHeap<QueuedState<StateType>>* priorityQueue, PriorityQueuesManager<StateType>* priorityQueuesManager, OffloadQueue<StateType>* offloadQueue);

template<typename ModelType, typename ProblemType, typename StateType>
void doOffloadCpuAsync(DD::MDD<ModelType, ProblemType, StateType> const * mdd, TS::Neighbourhood const * neighbourhood, OffloadQueue<StateType>* cpuQueue, Vector<std::thread>* cpuThreads, std::byte* scratchpadMem);

template<typename ModelType, typename ProblemType, typename StateType>
void doOffloadGpuAsync(DD::MDD<ModelType, ProblemType, StateType> const * mdd, TS::Neighbourhood const * neighbourhood, OffloadQueue<StateType>* gpuQueue);

template<typename ModelType, typename ProblemType, typename StateType>
__host__ __device__ void doOffload(DD::MDD<ModelType,ProblemType,StateType> const * mdd, TS::Neighbourhood const * neighbourhood, BB::OffloadedState<StateType>* offloadedState, std::byte* scratchpadMem);

template<typename ModelType, typename ProblemType, typename StateType>
__global__ void doOffloadKernel(DD::MDD<ModelType,ProblemType,StateType> const * mdd, TS::Neighbourhood const * neighbourhood, Vector<BB::OffloadedState<StateType>>* queue);

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
    unsigned int tabuLength = std::stoi(argv[8]);

    // *******************
    // Data initialization
    // *******************

    // Context initialization
    MallocType gpuMallocType = MallocType::Std;
    if (gpuMaxParallelism > 0)
    {
        gpuMallocType = MallocType::Managed;
        configGPU();
    };

    // Problems
    ProblemType* const cpuProblem = parseGrubHubInstance(problemFileName, MallocType::Std);
    ProblemType* const gpuProblem = parseGrubHubInstance(problemFileName, gpuMallocType);

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

    // Tabu Search
    memorySize = sizeof(Neighbourhood);
    memory = safeMalloc(memorySize, MallocType::Std);
    Neighbourhood* cpuTsNeighborhood = new (memory) Neighbourhood(cpuProblem, tabuLength, MallocType::Std);
    memory = safeMalloc(memorySize, MallocType::Std);
    Neighbourhood* cpuBbNeighborhood = new (memory) Neighbourhood(cpuProblem, tabuLength, MallocType::Std);
    Neighbourhood* cpuNeighborhood;
    memory = safeMalloc(memorySize, gpuMallocType);
    Neighbourhood* gpuTsNeighborhood = new (memory) Neighbourhood(gpuProblem, tabuLength, gpuMallocType);
    memory = safeMalloc(memorySize, gpuMallocType);
    Neighbourhood* gpuBbNeighborhood = new (memory) Neighbourhood(gpuProblem, tabuLength, gpuMallocType);
    Neighbourhood* gpuNeighborhood;

    // Queues
    PriorityQueuesManager<StateType> priorityQueuesManger(cpuProblem, queueMaxSize, 2);
    MaxHeap<QueuedState<StateType>> priorityQueue(hasSmallerCost<StateType>, queueMaxSize, MallocType::Std);
    priorityQueuesManger.registerQueue(&priorityQueue);

    // Offload
    memorySize = sizeof(OffloadQueue<StateType>);
    memory = safeMalloc(memorySize, MallocType::Std);
    OffloadQueue<StateType>* cpuOffloadQueue = new (memory) OffloadQueue<StateType>(cpuMdd, cpuMaxParallelism, MallocType::Std);
    memory = safeMalloc(memorySize, gpuMallocType);
    OffloadQueue<StateType>* gpuOffloadQueue = new (memory) OffloadQueue<StateType>(gpuMdd, gpuMaxParallelism, gpuMallocType);

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
    unsigned int tabuSearchIterationsCount = 0;
    unsigned int iterationsCount = 0;
    enum SearchStatus {BranchAndBound, TabuSearch} searchStatus = SearchStatus::BranchAndBound;
    unsigned int visitedStatesCount = 0;
    cpuNeighborhood = cpuBbNeighborhood;
    gpuNeighborhood = gpuBbNeighborhood;

    // ************
    // Begin search
    // ************


    // Enqueue root
    StateMetadata<StateType> const rootMetadata(0, StateType::MaxCost, root);
    priorityQueuesManger.enqueue(&rootMetadata);

    uint64_t searchStartTime = now();
    clearLine();
    printf("[INFO] Start branch and bound");
    printf(" | Time: ");
    printElapsedTime(now() - searchStartTime);
    printf(" | Iterations: %u\n", iterationsCount);
    do
    {
        switch(searchStatus)
        {
            case SearchStatus::TabuSearch:
            {
                if (priorityQueuesManger.isFull())
                {
                    clearLine();
                    printf("[INFO] Tabu search solution: ");
                    currentSolution->selectedValues.print(false);
                    printf(" | Value: %d\n", currentSolution->cost);

                    cpuNeighborhood->update(&currentSolution->selectedValues);
                    gpuNeighborhood->update(&currentSolution->selectedValues);

                    currentSolution->reset();
                    priorityQueuesManger.clearQueues();
                    priorityQueuesManger.enqueue(&rootMetadata);

                    tabuSearchIterationsCount += 1;

                    if(tabuSearchIterationsCount % 5 == 0)
                    {
                        cpuNeighborhood = cpuBbNeighborhood;
                        gpuNeighborhood = gpuBbNeighborhood;
                        currentSolution->reset();
                        priorityQueuesManger.clearQueues();
                        priorityQueuesManger.enqueue(&rootMetadata);
                        searchStatus = SearchStatus::BranchAndBound;

                        clearLine();
                        printf("[INFO] Start branch and bound search");
                        printf(" | Time: ");
                        printElapsedTime(now() - searchStartTime);
                        printf(" | Iterations: %u\n", iterationsCount);
                    }
                }
            }
            break;
            case SearchStatus::BranchAndBound:
            {
                if (priorityQueuesManger.isFull())
                {
                    clearLine();
                    printf("[INFO] Branch and bound solution: ");
                    currentSolution->selectedValues.print(false);
                    printf(" | Value: %d\n", currentSolution->cost);

                    cpuNeighborhood = cpuTsNeighborhood;
                    gpuNeighborhood = gpuTsNeighborhood;
                    cpuNeighborhood->update(&currentSolution->selectedValues);
                    gpuNeighborhood->update(&currentSolution->selectedValues);
                    currentSolution->reset();
                    priorityQueuesManger.clearQueues();
                    priorityQueuesManger.enqueue(&rootMetadata);
                    searchStatus = SearchStatus::TabuSearch;

                    clearLine();
                    printf("[INFO] Start tabu search");
                    printf(" | Time: ");
                    printElapsedTime(now() - searchStartTime);
                    printf(" | Iterations: %u\n", iterationsCount);
                }
            }
        }

        prepareOffload<StateType>(currentSolution->cost, &priorityQueue, &priorityQueuesManger, cpuOffloadQueue);
        prepareOffload<StateType>(currentSolution->cost, &priorityQueue, &priorityQueuesManger, gpuOffloadQueue);

        uint64_t cpuOffloadStartTime = now();
        uint64_t gpuOffloadStartTime = now();
        doOffloadCpuAsync(cpuMdd, cpuNeighborhood, cpuOffloadQueue, cpuThreads, scratchpadMem);
        doOffloadGpuAsync(gpuMdd, gpuNeighborhood, gpuOffloadQueue);
        waitOffloadCpu(cpuThreads);
        waitOffloadGpu();

        visitedStatesCount += cpuOffloadQueue->getSize();
        visitedStatesCount += gpuOffloadQueue->getSize();

        bool foundBetterSolution =
            checkForBetterSolutions(bestSolution, currentSolution, cpuOffloadQueue) or
            checkForBetterSolutions(bestSolution, currentSolution, gpuOffloadQueue);

        updatePriorityQueues(currentSolution->cost, &priorityQueuesManger, cpuOffloadQueue);
        updatePriorityQueues(currentSolution->cost, &priorityQueuesManger, gpuOffloadQueue);

        if (foundBetterSolution)
        {
            clearLine();
            printf("[INFO] Better solution found: ");
            bestSolution->selectedValues.print(false);
            printf(" | Value: %u", bestSolution->cost);
            printf(" | Time: ");
            printElapsedTime(now() - searchStartTime);
            printf(" | Iterations: %u", iterationsCount);
            printf(" | States to visit: %lu", priorityQueuesManger.getSize());
            printf(" | Visited states: %u\n", visitedStatesCount);
        }

        uint64_t cpuOffloadElapsedTime = max(1ul, now() - cpuOffloadStartTime);
        unsigned long int cpuSpeed = cpuOffloadQueue->getSize() * 1000 / cpuOffloadElapsedTime;
        uint64_t gpuOffloadElapsedTime = max(1ul, now() - gpuOffloadStartTime);
        unsigned long int gpuSpeed  = gpuOffloadQueue->getSize() * 1000 / gpuOffloadElapsedTime;
        clearLine();
        printf("[INFO] Time: ");
        printElapsedTime(now() - searchStartTime);
        printf(" | Value: %u", currentSolution->cost);
        printf(" | Iterations: %u", iterationsCount);
        printf(" | To Visit - Visited: %lu - %u", priorityQueuesManger.getSize(), visitedStatesCount);
        printf(" | CPU - GPU: %lu - %lu\r", cpuSpeed, gpuSpeed);
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
    printf(" | Visited: %u\n", visitedStatesCount);

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

OP::VRProblem * parseGrubHubInstance(char const * problemFileName, Memory::MallocType mallocType)
{
    // Parse instance
    std::ifstream problemFile(problemFileName);
    json problemJson;
    problemFile >> problemJson;

    // Init problem
    unsigned int const problemSize = sizeof(OP::VRProblem);
    std::byte* const memory = safeMalloc(problemSize, mallocType);
    unsigned int const variablesCount = problemJson["nodes"].size();
    OP::VRProblem* const problem  = new (memory) OP::VRProblem(variablesCount, mallocType);

    // Init variables
    new (problem->variables[0]) OP::Variable(0, 0);
    for(unsigned int variableIdx = 1; variableIdx < variablesCount - 1; variableIdx += 1)
    {
        new (problem->variables[variableIdx]) OP::Variable(2, variablesCount - 1);
    }
    new (problem->variables[variablesCount - 1]) OP::Variable(1, 1);

    // Init start/end locations
    problem->start = 0;
    problem->end = 1;

    // Init pickups and deliveries
    for(uint8_t pickup = 2; pickup < variablesCount; pickup += 2)
    {

        problem->pickups.incrementSize();
        *problem->pickups.back() = pickup;

        problem->deliveries.incrementSize();
        *problem->deliveries.back() = pickup + 1;
    }

    // Init distances
    for(unsigned int from = 0; from < variablesCount; from += 1)
    {
        for(unsigned int to = 0; to < variablesCount; to += 1)
        {
            *problem->distances[(from * variablesCount) + to] = problemJson["edges"][from][to];
        }
    }

    return problem;
}

template<typename StateType>
bool hasSmallerCost(QueuedState<StateType> const & queuedState0, QueuedState<StateType> const & queuedState1)
{
    unsigned int cost0 = queuedState0.state->cost;
    unsigned int cost1 = queuedState1.state->cost;

    return cost0 < cost1;
}

template<typename StateType>
bool hasBiggerCost(QueuedState<StateType> const & queuedState0, QueuedState<StateType> const & queuedState1)
{
    unsigned int cost0 = queuedState0.state->cost;
    unsigned int cost1 = queuedState1.state->cost;

    return cost0 > cost1;
}



template<typename StateType>
bool hasRandomPriority(QueuedState<StateType> const & queuedState0, QueuedState<StateType> const & queuedState1)
{
    return static_cast<bool>(rand() % 2);
}

template<typename StateType>
bool hasLessSelections(QueuedState<StateType> const & queuedState0, QueuedState<StateType> const & queuedState1)
{
    unsigned int selectedValuesCount0 = queuedState0.state->selectedValues.getSize();
    unsigned int selectedValuesCount1 = queuedState1.state->selectedValues.getSize();

    return selectedValuesCount0 < selectedValuesCount1;
}

template<typename StateType>
void updatePriorityQueues(unsigned int bestCost, PriorityQueuesManager<StateType>* priorityQueuesManager, OffloadQueue<StateType>* offloadQueue)
{
    for (OffloadedState<StateType>* offloadedState = offloadQueue->begin(); offloadedState !=  offloadQueue->end(); offloadedState += 1)
    {
        for (StateType* cutsetState = offloadedState->cutset.begin(); cutsetState != offloadedState->cutset.end(); cutsetState += 1)
        {
            StateMetadata<StateType> const cutsetStateMetadata(offloadedState->lowerbound, offloadedState->upperbound, cutsetState);
            if(boundsChk(bestCost, &cutsetStateMetadata))
            {
                if(not priorityQueuesManager->isFull())
                {
                    priorityQueuesManager->enqueue(&cutsetStateMetadata);
                }
            }
        };
    };
}

template<typename StateType>
bool boundsChk(unsigned int bestCost, StateMetadata<StateType> const * stateMetadata)
{
    return
        stateMetadata->lowerbound < stateMetadata->upperbound and
        stateMetadata->lowerbound < bestCost and
        stateMetadata->state->cost <= bestCost;
}

template<typename StateType>
bool checkForBetterSolutions(StateType* bestSolution, StateType* currentSolution, OffloadQueue<StateType>* offloadQueue)
{
    bool foundBetterSolution = false;

    for (OffloadedState<StateType>* offloadedState = offloadQueue->begin(); offloadedState != offloadQueue->end(); offloadedState += 1)
    {
        if (offloadedState->upperbound < bestSolution->cost)
        {
            *bestSolution = *offloadedState->upperboundState;
            foundBetterSolution = true;
        }

        if (offloadedState->upperbound < currentSolution->cost)
        {
            *currentSolution = *offloadedState->upperboundState;
        }
    };

    return foundBetterSolution;
}

template<typename StateType>
void  prepareOffload(unsigned int bestCost, MaxHeap<QueuedState<StateType>>* priorityQueue, PriorityQueuesManager<StateType>* priorityQueuesManager, OffloadQueue<StateType>* offloadQueue)
{
    offloadQueue->clear();
    while(not (priorityQueue->isEmpty() or offloadQueue->isFull()))
    {
        QueuedState<StateType> const * const queuedState = priorityQueue->front();
        if (boundsChk(bestCost, queuedState))
        {
            offloadQueue->enqueue(queuedState->state);
        }
        priorityQueuesManager->dequeue(queuedState->state);
    }
}

template<typename ModelType, typename ProblemType, typename StateType>
void doOffloadCpuAsync(MDD<ModelType, ProblemType, StateType> const * mdd, TS::Neighbourhood const * neighbourhood, OffloadQueue<StateType>* cpuQueue, Vector<std::thread>* cpuThreads, std::byte* scratchpadMem)
{
    if(not cpuQueue->isEmpty())
    {
        cpuThreads->clear();
        for (unsigned int offloadedStateIdx = 0; offloadedStateIdx < cpuQueue->getSize(); offloadedStateIdx += 1)
        {
            cpuThreads->incrementSize();
            new(cpuThreads->back()) std::thread(&doOffload<ModelType, ProblemType, StateType>, mdd, neighbourhood, cpuQueue->at(offloadedStateIdx), &scratchpadMem[mdd->scratchpadMemSize * offloadedStateIdx]);
        }
    }
}

template<typename ModelType, typename ProblemType, typename StateType>
void doOffloadGpuAsync(MDD<ModelType, ProblemType, StateType> const * mdd, TS::Neighbourhood const * neighbourhood, OffloadQueue<StateType>* gpuQueue)
{
    if(not gpuQueue->isEmpty())
    {
        doOffloadKernel<ModelType, ProblemType, StateType><<<gpuQueue->getSize(), 1, mdd->scratchpadMemSize>>>(mdd, neighbourhood, gpuQueue);
    }
}

template<typename ModelType, typename ProblemType, typename StateType>
__host__ __device__
void doOffload(MDD<ModelType,ProblemType,StateType> const * mdd, TS::Neighbourhood const * neighbourhood, OffloadedState<StateType>* offloadedState, std::byte* scratchpadMem)
{
    //Preparation to build MDDs
    StateType const * top = offloadedState->state;
    LightVector<StateType>* const cutset = &offloadedState->cutset;
    StateType * const bottom = offloadedState->upperboundState;

    //Build MDDs
    mdd->buildTopDown(MDD<ModelType,ProblemType,StateType>::Type::Relaxed, top, cutset, bottom, neighbourhood, scratchpadMem);
    offloadedState->lowerbound = bottom->cost;
    mdd->buildTopDown(MDD<ModelType,ProblemType,StateType>::Type::Restricted, top, cutset, bottom, neighbourhood, scratchpadMem);
    offloadedState->upperbound = bottom->cost;
}

template<typename ModelType, typename ProblemType, typename StateType>
__global__
void doOffloadKernel(DD::MDD<ModelType,ProblemType,StateType> const * mdd, TS::Neighbourhood const * neighbourhood, OffloadQueue<StateType>* gpuQueue)
{
    extern __shared__ unsigned int sharedMem[];
    std::byte* scratchpadMem = reinterpret_cast<std::byte*>(sharedMem);

    if(threadIdx.x == 0)
    {
        OffloadedState<StateType>* offloadedState = gpuQueue->at(blockIdx.x);
        doOffload(mdd, neighbourhood, offloadedState, scratchpadMem);
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
