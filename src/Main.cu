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

#include "BB/OffloadQueue.cuh"
#include "BB/PriorityQueuesManager.cuh"
#include "DD/MDD.cuh"
#include "DP/VRPModel.cuh"
#include "OP/VRProblem.cuh"

using namespace std;
using json = nlohmann::json;
using namespace Memory;
using namespace Chrono;
using namespace BB;
using namespace DD;
using namespace DP;
using namespace OP;
using ModelType = VRPModel;
using ProblemType = VRProblem;
using StateType = VRPState;

// Auxiliary functions
void configGPU();
OP::VRProblem* parseGrubHubInstance(char const * problemFileName, Memory::MallocType mallocType);

// Comparators
template<typename QueuedStateType>
bool hasBiggerCost(QueuedStateType const & queuedState0, QueuedStateType const & queuedState1);

template<typename QueuedStateType>
bool hasSmallerCost(QueuedStateType const & queuedState0, QueuedStateType const & queuedState1);

// Queues
template<typename StateType>
bool boundsChk(unsigned int bestCost, StateMetadata<StateType> const * stateMetadata);

template<typename StateType>
void updatePriorityQueues(unsigned int bestCost, PriorityQueuesManager<StateType>* priorityQueuesManager, OffloadQueue<StateType>* offloadQueue);

// Search
template<typename StateType>
bool checkForBetterSolutions(StateType* bestSolution, OffloadQueue<StateType>* offloadQueue);

// Offload
template<typename StateType>
void prepareOffload(unsigned int bestCost, MaxHeap<QueuedState<StateType>>* priorityQueue, PriorityQueuesManager<StateType>* priorityQueuesManager, OffloadQueue<StateType>* offloadQueue);

template<typename ModelType, typename ProblemType, typename StateType>
void doOffloadCpu(DD::MDD<ModelType,ProblemType,StateType> const * mdd, OffloadQueue<StateType>* cpuQueue, std::byte* scratchpadMem);

template<typename ModelType, typename ProblemType, typename StateType>
void doOffloadGpu(DD::MDD<ModelType,ProblemType,StateType> const * mdd, OffloadQueue<StateType>* gpuQueue);

template<typename ModelType, typename ProblemType, typename StateType>
__host__ __device__ void doOffload(DD::MDD<ModelType,ProblemType,StateType> const * mdd, BB::OffloadedState<StateType>* offloadedState, std::byte* scratchpadMem);

template<typename ModelType, typename ProblemType, typename StateType>
__global__ void doOffloadKernel(DD::MDD<ModelType,ProblemType,StateType> const * mdd, Vector<BB::OffloadedState<StateType>>* queue);

// Debug
void printElapsedTime(uint64_t elapsedTimeMs);

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

    // Context initialization
    MallocType gpuDataMallocType = MallocType::Std;
    if (gpuMaxParallelism > 0)
    {
        gpuDataMallocType = MallocType::Managed;
        configGPU();
    };

    // Problem
    ProblemType const * const problem = parseGrubHubInstance(problemFileName, gpuDataMallocType);

    // Model
    unsigned int memorySize = sizeof(ModelType);
    std::byte* memory = safeMalloc(memorySize, gpuDataMallocType);
    ModelType * const model = new (memory) ModelType(problem);

    // MDDs
    memorySize = sizeof(MDD<ModelType,ProblemType,StateType>);
    memory = safeMalloc(memorySize, MallocType::Std);
    MDD<ModelType,ProblemType,StateType>* const cpuMdd = new (memory) MDD<ModelType,ProblemType,StateType>(model, cpuMaxWidth);

    memorySize = sizeof(MDD<ModelType,ProblemType,StateType>);
    memory = safeMalloc(memorySize, gpuDataMallocType);
    MDD<ModelType,ProblemType,StateType>* const gpuMdd = new (memory) MDD<ModelType,ProblemType,StateType>(model, gpuMaxWidth);

    // Context initialization
    std::byte* scratchpadMem = nullptr;
    if(cpuMaxParallelism > 0)
    {
        scratchpadMem = safeMalloc(cpuMdd->scratchpadMemSize * cpuMaxParallelism, MallocType::Std);
    }

    // Queues
    PriorityQueuesManager<StateType> priorityQueuesManger(problem, queueMaxSize, 2);
    MaxHeap<QueuedState<StateType>> cpuPriorityQueue(hasSmallerCost<QueuedState<StateType>>, queueMaxSize, MallocType::Std);
    priorityQueuesManger.registerQueue(&cpuPriorityQueue);
    MaxHeap<QueuedState<StateType>> gpuPriorityQueue(hasBiggerCost<QueuedState<StateType>>, queueMaxSize, MallocType::Std);
    priorityQueuesManger.registerQueue(&gpuPriorityQueue);

    // Offload queues
    OffloadQueue<StateType> cpuOffloadQueue(cpuMdd, cpuMaxParallelism, MallocType::Std);
    OffloadQueue<StateType> gpuOffloadQueue(gpuMdd, gpuMaxParallelism, gpuDataMallocType);

    // Best solution
    unsigned int const stateSize = sizeof(StateType);
    std::byte* stateMem = safeMalloc(stateSize, gpuDataMallocType);
    memory = StateType::mallocStorages(problem, 1, gpuDataMallocType);
    StateType* bestSolution = new (stateMem) StateType(problem, memory);
    bestSolution->cost = StateType::MaxCost;

    // Root
    stateMem = safeMalloc(stateSize, gpuDataMallocType);
    memory = StateType::mallocStorages(problem, 1, gpuDataMallocType);
    StateType* root = new (stateMem) StateType(problem, memory);
    model->makeRoot(root);

    // Enqueue root
    StateMetadata<StateType> const rootMetadata(0, StateType::MaxCost, root);
    priorityQueuesManger.enqueue(&rootMetadata);

    // Search
    unsigned int visitedStatesCount = 0;
    unsigned int iterationsCount = 0;
    uint64_t searchStartTime = now();
    do
    {
        prepareOffload<StateType>(bestSolution->cost, &cpuPriorityQueue, &priorityQueuesManger, &cpuOffloadQueue);
        prepareOffload<StateType>(bestSolution->cost, &gpuPriorityQueue, &priorityQueuesManger, &gpuOffloadQueue);

        uint64_t cpuOffloadStartTime = now();
        doOffloadCpu(cpuMdd, &cpuOffloadQueue, scratchpadMem);
        visitedStatesCount += cpuOffloadQueue.getSize();

        uint64_t gpuOffloadStartTime = now();
        doOffloadGpu(gpuMdd, &gpuOffloadQueue);
        visitedStatesCount += gpuOffloadQueue.getSize();

        bool foundBetterSolution =
            checkForBetterSolutions(bestSolution, &cpuOffloadQueue) or
            checkForBetterSolutions(bestSolution, &gpuOffloadQueue);

        updatePriorityQueues(bestSolution->cost, &priorityQueuesManger, &cpuOffloadQueue);
        updatePriorityQueues(bestSolution->cost, &priorityQueuesManger, &gpuOffloadQueue);

        if(foundBetterSolution)
        {
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
            if (cpuOffloadQueue.getSize() > 0 )
            {
                uint64_t cpuOffloadElapsedTime = max(1ul, now() - cpuOffloadStartTime);
                cpuSpeed = cpuOffloadQueue.getSize() * 1000 / cpuOffloadElapsedTime;
            }
            unsigned long int gpuSpeed = 0;
            if (gpuOffloadQueue.getSize() > 0 )
            {
                uint64_t gpuOffloadElapsedTime = max(1ul, now() - gpuOffloadStartTime);
                gpuSpeed = gpuOffloadQueue.getSize() * 1000 / gpuOffloadElapsedTime;
            }
            printf("[INFO] CPU Speed: %5lu states/s", cpuSpeed);
            printf(" | GPU Speed: %5lu states/s", gpuSpeed);
            printf(" | Time: ");
            printElapsedTime(now() - searchStartTime);
            printf(" | Iterations: %u", iterationsCount);
            printf(" | State to visit: %u", priorityQueuesManger.getQueuesSize());
            printf(" | Visited states: %u\r", visitedStatesCount);
        }

        iterationsCount += 1;
    }
    while(not (priorityQueuesManger.areQueuesEmpty() or (now() - searchStartTime) > timeoutSeconds * 1000));

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

    printf("[INFO] Problem: %s | Locations: %d | Pickups/Deliveries: %lu\n", problemFileName, variablesCount, problem->pickups.getSize());

    return problem;
}

template<typename QueuedStateType>
bool hasBiggerCost(QueuedStateType const & queuedState0, QueuedStateType const & queuedState1)
{
    return queuedState0.state->cost > queuedState1.state->cost;
}

template<typename QueuedStateType>
bool hasSmallerCost(QueuedStateType const & queuedState0, QueuedStateType const & queuedState1)
{
    return queuedState0.state->cost < queuedState1.state->cost;
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
                priorityQueuesManager->enqueue(&cutsetStateMetadata);
            }
        };
    };
}

template<typename StateType>
bool boundsChk(unsigned int bestCost, StateMetadata<StateType> const * stateMetadata)
{
    return true or
        stateMetadata->lowerbound < stateMetadata->upperbound and
        stateMetadata->lowerbound < bestCost and
        stateMetadata->state->cost < bestCost;
}

template<typename StateType>
bool checkForBetterSolutions(StateType* bestSolution, OffloadQueue<StateType>* offloadQueue)
{
    bool foundBetterSolution = false;

    for (OffloadedState<StateType>* offloadedState = offloadQueue->begin(); offloadedState != offloadQueue->end(); offloadedState += 1)
    {
        if (offloadedState->upperbound < bestSolution->cost)
        {
            *bestSolution = *offloadedState->upperboundState;
            foundBetterSolution = true;
        }
    };

    return foundBetterSolution;
}

template<typename StateType>
void prepareOffload(unsigned int bestCost, MaxHeap<QueuedState<StateType>>* priorityQueue, PriorityQueuesManager<StateType>* priorityQueuesManager, OffloadQueue<StateType>* offloadQueue)
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
void doOffloadCpu(MDD<ModelType,ProblemType,StateType> const * mdd, OffloadQueue<StateType>* cpuQueue, std::byte* scratchpadMem)
{
    for(unsigned int offloadedStateIdx = 0; offloadedStateIdx < cpuQueue->getSize(); offloadedStateIdx += 1)
    {
        doOffload(mdd, cpuQueue->at(offloadedStateIdx), &scratchpadMem[mdd->scratchpadMemSize * offloadedStateIdx]);
    }
}

template<typename ModelType, typename ProblemType, typename StateType>
void doOffloadGpu(MDD<ModelType,ProblemType,StateType> const * mdd, OffloadQueue<StateType>* gpuQueue)
{
    doOffloadKernel<ModelType,ProblemType,StateType><<<gpuQueue->getSize(), 1, mdd->scratchpadMemSize>>>(mdd, gpuQueue);
    cudaDeviceSynchronize();
}

template<typename ModelType, typename ProblemType, typename StateType>
__host__ __device__
void doOffload(MDD<ModelType,ProblemType,StateType> const * mdd, OffloadedState<StateType>* offloadedState, std::byte* scratchpadMem)
{
    //Preparation to build MDDs
    StateType const * top = offloadedState->state;
    LightVector<StateType>* const cutset = &offloadedState->cutset;
    StateType * const bottom = offloadedState->upperboundState;

    //Build MDDs
    mdd->buildTopDown(MDD<ModelType,ProblemType,StateType>::Type::Relaxed, top, cutset, bottom, scratchpadMem);
    offloadedState->lowerbound = bottom->cost;
    mdd->buildTopDown(MDD<ModelType,ProblemType,StateType>::Type::Restricted, top, cutset, bottom, scratchpadMem);
    offloadedState->upperbound = bottom->cost;
}

template<typename ModelType, typename ProblemType, typename StateType>
__global__
void doOffloadKernel(DD::MDD<ModelType,ProblemType,StateType> const * mdd, OffloadQueue<StateType>* gpuQueue)
{
    extern __shared__ unsigned int sharedMem[];
    std::byte* scratchpadMem = reinterpret_cast<std::byte*>(sharedMem);

    if(blockIdx.x * blockDim.x + threadIdx.x == 0)
    {
        BB::OffloadedState<StateType>* offloadedState = gpuQueue->at(blockIdx.x);
        doOffload(mdd, offloadedState, scratchpadMem);
    };
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