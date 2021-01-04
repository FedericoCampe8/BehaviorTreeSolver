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

#include "BB/OffloadQueue.cuh"
#include "BB/PriorityQueuesManager.cuh"
#include "DD/MDD.cuh"
#include "DP/VRPModel.cuh"
#include "LNS/Neighbourhood.cuh"

using namespace std;
using json = nlohmann::json;
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
bool checkForBetterSolutions(StateType* bestSolution, OffloadQueue<StateType>* offloadQueue);

template<typename StateType>
void updateLnsSolutions(StateType* lnsSolution, OffloadQueue<StateType>* offloadQueue);

// Offload
template<typename StateType>
void prepareOffload(unsigned int bestCost, MaxHeap<QueuedState<StateType>>* priorityQueue, PriorityQueuesManager<StateType>* priorityQueuesManager, OffloadQueue<StateType>* offloadQueue);

template<typename ModelType, typename ProblemType, typename StateType>
void doOffloadCpuAsync(DD::MDD<ModelType, ProblemType, StateType> const * mdd, LNS::Neighbourhood const * neighbourhood, OffloadQueue<StateType>* cpuQueue, Vector<std::thread>* cpuThreads, std::byte* scratchpadMem);

template<typename ModelType, typename ProblemType, typename StateType>
void doOffloadGpuAsync(DD::MDD<ModelType, ProblemType, StateType> const * mdd, LNS::Neighbourhood const * neighbourhood, OffloadQueue<StateType>* gpuQueue);

template<typename ModelType, typename ProblemType, typename StateType>
__host__ __device__ void doOffload(DD::MDD<ModelType,ProblemType,StateType> const * mdd, LNS::Neighbourhood const * neighbourhood, BB::OffloadedState<StateType>* offloadedState, std::byte* scratchpadMem);

template<typename ModelType, typename ProblemType, typename StateType>
__global__ void doOffloadKernel(DD::MDD<ModelType,ProblemType,StateType> const * mdd, LNS::Neighbourhood const * neighbourhood, Vector<BB::OffloadedState<StateType>>* queue);

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
    unsigned int const lnsFixPercentage = std::stoi(argv[8]);
    unsigned int const lnsQueueMaxSize = std::stoi(argv[9]);
    unsigned int randomSeed = std::stoi(argv[10]);
    unsigned int notBetterSolutionPercentage = std::stoi(argv[11]);
    unsigned int randomStatePercentage = std::stoi(argv[12]);

    // *******************
    // Data initialization
    // *******************

    // Context initialization
    std::mt19937 rng(randomSeed);
    MallocType gpuMallocTime = MallocType::Std;
    if (gpuMaxParallelism > 0)
    {
        gpuMallocTime = MallocType::Managed;
        configGPU();
    };

    // Problems
    ProblemType* const cpuProblem = parseGrubHubInstance(problemFileName, MallocType::Std);
    ProblemType* const gpuProblem = parseGrubHubInstance(problemFileName, gpuMallocTime);

    // Models
    unsigned int memorySize = sizeof(ModelType);
    std::byte* memory = safeMalloc(memorySize, MallocType::Std);
    ModelType * const cpuModel = new (memory) ModelType(cpuProblem);
    memory = safeMalloc(memorySize, gpuMallocTime);
    ModelType * const gpuModel = new (memory) ModelType(gpuProblem);

    // MDDs
    memorySize = sizeof(MDD<ModelType,ProblemType,StateType>);
    memory = safeMalloc(memorySize, MallocType::Std);
    MDD<ModelType,ProblemType,StateType>* const cpuMdd = new (memory) MDD<ModelType,ProblemType,StateType>(cpuModel, cpuMaxWidth);
    memory = safeMalloc(memorySize, gpuMallocTime);
    MDD<ModelType,ProblemType,StateType>* const gpuMdd = new (memory) MDD<ModelType,ProblemType,StateType>(gpuModel, gpuMaxWidth);

    // Context initialization
    Vector<std::thread>* cpuThreads = new Vector<std::thread>(cpuMaxParallelism, MallocType::Std);
    std::byte* scratchpadMem = safeMalloc(cpuMdd->scratchpadMemSize * cpuMaxParallelism, MallocType::Std);

    // LSN
    memorySize = sizeof(Neighbourhood);
    memory = safeMalloc(memorySize, MallocType::Std);
    Neighbourhood* cpuNeighborhood = new (memory) Neighbourhood(cpuProblem, MallocType::Std);
    memory = safeMalloc(memorySize, gpuMallocTime);
    Neighbourhood* gpuNeighborhood = new (memory) Neighbourhood(gpuProblem, gpuMallocTime);

    memorySize = sizeof(StateType);
    memory = safeMalloc(memorySize, MallocType::Std);
    StateType* lnsSolution = new (memory) StateType(cpuProblem, StateType::mallocStorages(cpuProblem, 1, MallocType::Std));
    lnsSolution->cost = StateType::MaxCost;

    // Queues
    PriorityQueuesManager<StateType> priorityQueuesManger(cpuProblem, queueMaxSize, 2);
    MaxHeap<QueuedState<StateType>> searchPriorityQueue(hasSmallerCost<StateType>, queueMaxSize, MallocType::Std);
    priorityQueuesManger.registerQueue(&searchPriorityQueue);
    MaxHeap<QueuedState<StateType>> randomPriorityQueue(hasBiggerCost<StateType>, queueMaxSize, MallocType::Std);
    priorityQueuesManger.registerQueue(&randomPriorityQueue);

    // Offload
    memorySize = sizeof(OffloadQueue<StateType>);
    memory = safeMalloc(memorySize, MallocType::Std);
    OffloadQueue<StateType>* cpuOffloadQueue = new (memory) OffloadQueue<StateType>(cpuMdd, cpuMaxParallelism, MallocType::Std);
    memory = safeMalloc(memorySize, gpuMallocTime);
    OffloadQueue<StateType>* gpuOffloadQueue = new (memory) OffloadQueue<StateType>(gpuMdd, gpuMaxParallelism, gpuMallocTime);

    // Best solution
    memorySize = sizeof(StateType);
    memory = safeMalloc(memorySize, MallocType::Std);
    StateType* bestSolution = new (memory) StateType(cpuProblem, StateType::mallocStorages(cpuProblem, 1, MallocType::Std));
    bestSolution->cost = StateType::MaxCost;

    // Root
    memorySize = sizeof(StateType);
    memory = safeMalloc(memorySize, MallocType::Std);
    StateType* root = new (memory) StateType(cpuProblem, StateType::mallocStorages(cpuProblem, 1, MallocType::Std));
    cpuModel->makeRoot(root);

    // Search
    unsigned int iterationsCount = 0;
    enum SearchStatus {InitialSolution, LNS} searchStatus;
    unsigned int visitedStatesCount = 0;


    // ************
    // Begin search
    // ************

    // Init context
    std::uniform_int_distribution<unsigned int> randomDistribution(0,100);
    searchStatus = SearchStatus::InitialSolution;

    // Enqueue root
    StateMetadata<StateType> const rootMetadata(0, StateType::MaxCost, root);
    priorityQueuesManger.enqueue(&rootMetadata);

    uint64_t searchStartTime = now();
    do
    {
        if (searchStatus == SearchStatus::InitialSolution and priorityQueuesManger.isFull())
        {
            searchStatus = SearchStatus::LNS;
            lnsSolution->cost = StateType::MaxCost;

            clearLine();
            printf("[INFO] Switching to LNS search");
            printf(" | Time: ");
            printElapsedTime(now() - searchStartTime);
            printf(" | Iterations: %u", iterationsCount);
            printf(" | States to visit: %lu", priorityQueuesManger.getSize());
            printf(" | Visited states: %u\n", visitedStatesCount);

        }

        if (searchStatus == SearchStatus::LNS and (priorityQueuesManger.isEmpty() or priorityQueuesManger.getSize() > lnsQueueMaxSize or priorityQueuesManger.isFull()))
        {
            priorityQueuesManger.clearQueues();
            priorityQueuesManger.enqueue(&rootMetadata);

            cpuNeighborhood->reset();
            if (lnsSolution->cost < bestSolution->cost)
            {
                printf("[INFO] Worst solution found: ");
                lnsSolution->selectedValues.print();
            }
            if (randomDistribution(rng) < notBetterSolutionPercentage)
            {
                cpuNeighborhood->fixVariables(&lnsSolution->selectedValues, lnsFixPercentage, &rng);
            }
            else
            {
                cpuNeighborhood->fixVariables(&bestSolution->selectedValues, lnsFixPercentage, &rng);
            }
            lnsSolution->cost = StateType::MaxCost;

            *gpuNeighborhood = *cpuNeighborhood;
        }

        MaxHeap<QueuedState<StateType>>* priorityQueue = &searchPriorityQueue;
        if (searchStatus == SearchStatus::LNS and randomDistribution(rng) < randomStatePercentage)
        {
            priorityQueue = &randomPriorityQueue;
        }

        prepareOffload<StateType>(lnsSolution->cost, priorityQueue, &priorityQueuesManger, cpuOffloadQueue);
        prepareOffload<StateType>(lnsSolution->cost, priorityQueue, &priorityQueuesManger, gpuOffloadQueue);


        uint64_t cpuOffloadStartTime = now();
        doOffloadCpuAsync(cpuMdd, cpuNeighborhood, cpuOffloadQueue, cpuThreads, scratchpadMem);

        uint64_t gpuOffloadStartTime = now();
        doOffloadGpuAsync(gpuMdd, gpuNeighborhood, gpuOffloadQueue);

        waitOffloadCpu(cpuThreads);
        waitOffloadGpu();

        visitedStatesCount += cpuOffloadQueue->getSize();
        visitedStatesCount += gpuOffloadQueue->getSize();

        bool foundBetterSolution =
                checkForBetterSolutions(bestSolution, cpuOffloadQueue) or
                checkForBetterSolutions(bestSolution, gpuOffloadQueue);

        updateLnsSolutions(lnsSolution, cpuOffloadQueue);
        updateLnsSolutions(lnsSolution, gpuOffloadQueue);

        updatePriorityQueues(bestSolution->cost, &priorityQueuesManger, cpuOffloadQueue);
        updatePriorityQueues(bestSolution->cost, &priorityQueuesManger, gpuOffloadQueue);

        if(foundBetterSolution)
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
        else
        {
            unsigned long int cpuSpeed = 0;
            if (cpuOffloadQueue->getSize() > 0)
            {
                uint64_t cpuOffloadElapsedTime = max(1ul, now() - cpuOffloadStartTime);
                cpuSpeed = cpuOffloadQueue->getSize() * 1000 / cpuOffloadElapsedTime;
            }

            unsigned long int gpuSpeed = 0;
            if (gpuOffloadQueue->getSize() > 0)
            {
                uint64_t gpuOffloadElapsedTime = max(1ul, now() - gpuOffloadStartTime);
                gpuSpeed = gpuOffloadQueue->getSize() * 1000 / gpuOffloadElapsedTime;
            }

            clearLine();
            printf("[INFO] Time: ");
            printElapsedTime(now() - searchStartTime);
            if (searchStatus == SearchStatus::LNS )
            {
                printf(" | Neighborhood: ");
                cpuNeighborhood->print(false);
            }
            printf(" | CPU Speed: %5lu states/s", cpuSpeed);
            printf(" | GPU Speed: %5lu states/s", gpuSpeed);
            printf(" | Iterations: %u", iterationsCount);
            printf(" | State to visit: %lu", priorityQueuesManger.getSize());
            printf(" | Visited states: %u\r", visitedStatesCount);
            fflush(stdout);

        }

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
void updateLnsSolutions(StateType* lnsSolution, OffloadQueue<StateType>* offloadQueue)
{
    for (OffloadedState<StateType>* offloadedState = offloadQueue->begin(); offloadedState != offloadQueue->end(); offloadedState += 1)
    {
        if (offloadedState->upperbound < lnsSolution->cost)
        {
            *lnsSolution = *offloadedState->upperboundState;
        }
    };
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
void doOffloadCpuAsync(MDD<ModelType, ProblemType, StateType> const * mdd, LNS::Neighbourhood const * neighbourhood, OffloadQueue<StateType>* cpuQueue, Vector<std::thread>* cpuThreads, std::byte* scratchpadMem)
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
void doOffloadGpuAsync(MDD<ModelType, ProblemType, StateType> const * mdd, LNS::Neighbourhood const * neighbourhood, OffloadQueue<StateType>* gpuQueue)
{
    if(not gpuQueue->isEmpty())
    {
        doOffloadKernel<ModelType, ProblemType, StateType><<<gpuQueue->getSize(), 1, mdd->scratchpadMemSize>>>(mdd, neighbourhood, gpuQueue);
    }
}

template<typename ModelType, typename ProblemType, typename StateType>
__host__ __device__
void doOffload(MDD<ModelType,ProblemType,StateType> const * mdd, LNS::Neighbourhood const * neighbourhood, OffloadedState<StateType>* offloadedState, std::byte* scratchpadMem)
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
void doOffloadKernel(DD::MDD<ModelType,ProblemType,StateType> const * mdd, LNS::Neighbourhood const * neighbourhood, OffloadQueue<StateType>* gpuQueue)
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
