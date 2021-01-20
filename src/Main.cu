#include <algorithm>
#include <thread>
#include <Utils/Chrono.cuh>
#include "DP/VRPModel.cuh"
#include "OffloadBuffer.cuh"
#include "BB/PriorityQueue.cuh"

using namespace std;
using namespace Memory;
using namespace Chrono;
using namespace BB;
using namespace DD;
using namespace DP;
using namespace OP;
using ProblemType = VRProblem;
using StateType = VRPState;

// Auxiliary functions
void configGPU();

// Search
template<typename ProblemType, typename StateType>
void updatePriorityQueue(StateType* bestSolution, unsigned int * filteredStatesCount, PriorityQueue<StateType>* priorityQueue, OffloadBuffer<ProblemType, StateType>* offloadBuffer);

template<typename StateType>
bool boundsCheck(StateType* bestSolution, AugmentedState<StateType> const * augmentedState);

template<typename ProblemType, typename StateType>
bool checkForBetterSolutions(StateType* bestSolution, StateType* currentSolution, OffloadBuffer<ProblemType,StateType>* offloadBuffer);

// Offload
template<typename ProblemType, typename StateType>
void prepareOffload(StateType* bestSolution, unsigned int * filteredStatesCount, PriorityQueue<StateType>* priorityQueue, OffloadBuffer<ProblemType, StateType>* offloadBuffer);

template<typename ProblemType, typename StateType>
void prepareOffload(AugmentedState<StateType> const * augmentedState, OffloadBuffer<ProblemType,StateType>* offloadBuffer);

template<typename ProblemType, typename StateType>
void doOffloadCpuAsync(OffloadBuffer<ProblemType,StateType>* cpuOffloadBuffer, Vector<std::thread>* cpuThreads);

template<typename ProblemType, typename StateType>
void doOffloadGpuAsync(OffloadBuffer<ProblemType,StateType>* gpuOffloadBuffer);

template<typename ProblemType, typename StateType>
__global__ void doOffloadKernel(OffloadBuffer<ProblemType,StateType>* offloadBuffer);

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
    Vector<std::thread>* cpuThreads = new Vector<std::thread>(cpuMaxParallelism, MallocType::Std);

    // Problems
    ProblemType* const cpuProblem = VRProblem::parseGrubHubInstance(problemFileName, MallocType::Std);
    ProblemType* const gpuProblem = VRProblem::parseGrubHubInstance(problemFileName, gpuMallocType);

    // PriorityQueue
    PriorityQueue<StateType> priorityQueue(cpuProblem, queueMaxSize);

    // Offload
    unsigned int memorySize = sizeof(OffloadBuffer<ProblemType,StateType>);
    byte* memory = safeMalloc(memorySize, MallocType::Std);
    OffloadBuffer<ProblemType,StateType>* cpuOffloadBuffer = new (memory) OffloadBuffer<ProblemType,StateType>(cpuProblem, cpuMaxWidth, cpuMaxParallelism, MallocType::Std);
    memory = safeMalloc(memorySize, gpuMallocType);
    OffloadBuffer<ProblemType,StateType>* gpuOffloadBuffer = new (memory) OffloadBuffer<ProblemType,StateType>(gpuProblem, gpuMaxWidth, gpuMaxParallelism, gpuMallocType);

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
    priorityQueue.insert(root);

    // Search
    unsigned int iterationsCount = 0;
    enum SearchStatus {BB, LNS} searchStatus = SearchStatus::BB;
    unsigned int visitedStatesCount = 0;
    unsigned int filteredStatesCount = 0;

    // ************
    // Begin search
    // ************
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

                prepareOffload(bestSolution, &filteredStatesCount, &priorityQueue, cpuOffloadBuffer);
                prepareOffload(bestSolution, &filteredStatesCount, &priorityQueue, gpuOffloadBuffer);
            }
                break;
            case SearchStatus::LNS:
            {
                clearLine();
                printf("[INFO] Large neighborhood search");

                prepareOffload(&augmentedRoot, cpuOffloadBuffer);
                prepareOffload(&augmentedRoot, gpuOffloadBuffer);
                cpuOffloadBuffer->generateNeighbourhoods(currentSolution, lnsEqPercentage, lnsNeqPercentage, &rng);
                gpuOffloadBuffer->generateNeighbourhoods(currentSolution, lnsEqPercentage, lnsNeqPercentage, &rng);
                currentSolution->reset();
            }
                break;
        }

        uint64_t cpuOffloadStartTime = now();
        doOffloadCpuAsync(cpuOffloadBuffer, cpuThreads);

        uint64_t gpuOffloadStartTime = now();
        doOffloadGpuAsync(gpuOffloadBuffer);

        waitOffloadCpu(cpuThreads);
        waitOffloadGpu();

        visitedStatesCount += cpuOffloadBuffer->getSize();
        visitedStatesCount += gpuOffloadBuffer->getSize();

        bool foundBetterSolution =
                checkForBetterSolutions(bestSolution, currentSolution, cpuOffloadBuffer) or
                checkForBetterSolutions(bestSolution, currentSolution, gpuOffloadBuffer);

        updatePriorityQueue(bestSolution, &filteredStatesCount, &priorityQueue, cpuOffloadBuffer);
        updatePriorityQueue(bestSolution, &filteredStatesCount, &priorityQueue, gpuOffloadBuffer);

        if(foundBetterSolution)
        {
            clearLine();
            printf("[INFO] Better solution found: ");
            bestSolution->selectedValues.print(false);
            printf(" | Value: %u", bestSolution->cost);
            printf(" | Time: ");
            printElapsedTime(now() - searchStartTime);
            printf(" | Iterations: %u", iterationsCount);
            printf(" | States: %u - %u - %u\n", visitedStatesCount, priorityQueue.getSize(), filteredStatesCount);
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
            printf(" | States: %u - %u - %u", visitedStatesCount, priorityQueue.getSize(), filteredStatesCount);
            printf(" | Speed: %lu - %lu\r", cpuSpeed, gpuSpeed);
        }
        fflush(stdout);
        iterationsCount += 1;
    }
    while(now() - searchStartTime < timeoutSeconds * 1000 and (not priorityQueue.isEmpty()));

    clearLine();
    printf("[RESULT] Solution: ");
    bestSolution->selectedValues.print(false);
    printf(" | Value: %u", bestSolution->cost);
    printf(" | Time: ");
    printElapsedTime(now() - searchStartTime);
    printf(" | Iterations: %u", iterationsCount);
    printf(" | States: %u - %u - %u\n", visitedStatesCount, priorityQueue.getSize(), filteredStatesCount);

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
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
}

template<typename ProblemType, typename StateType>
void updatePriorityQueue(StateType* bestSolution, unsigned int * filteredStatesCount, PriorityQueue<StateType>* priorityQueue, OffloadBuffer<ProblemType, StateType>* offloadBuffer)
{
    for (unsigned int index = 0; index < offloadBuffer->getSize(); index += 1)
    {
        if(boundsCheck(bestSolution, offloadBuffer->getAugmentedState(index)))
        {
            Vector<StateType> const* const cutset = offloadBuffer->getMDD(index)->getCutset();
            for (StateType* cutsetState = cutset->begin(); cutsetState != cutset->end(); cutsetState += 1)
            {

                if (not priorityQueue->isFull())
                {
                    priorityQueue->insert(cutsetState);
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

    for (unsigned int index = 0; index < offloadBuffer->getSize(); index += 1)
    {
        StateType const * const approximateSolution = offloadBuffer->getMDD(index)->getBottom();
        if (approximateSolution->cost < currentSolution->cost)
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
            priorityQueue->popFront();
        }
        else
        {
            *filteredStatesCount += 1;
        }
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
void doOffloadCpuAsync(OffloadBuffer<ProblemType,StateType>* cpuOffloadBuffer, Vector<std::thread>* cpuThreads)
{
    cpuThreads->clear();
    for (unsigned int index = 0; index < cpuOffloadBuffer->getSize(); index += 1)
    {
        cpuThreads->resize(cpuThreads->getSize() + 1);
        new (cpuThreads->back()) std::thread(&OffloadBuffer<ProblemType,StateType>::doOffload,cpuOffloadBuffer,index);
    }
}

template<typename ProblemType, typename StateType>
void doOffloadGpuAsync(OffloadBuffer<ProblemType,StateType>* gpuOffloadBuffer)
{
    if(not gpuOffloadBuffer->isEmpty())
    {
        doOffloadKernel<ProblemType, StateType><<<gpuOffloadBuffer->getSize(), 1>>>(gpuOffloadBuffer);
    }
}

template<typename ProblemType, typename StateType>
__global__
void doOffloadKernel(OffloadBuffer<ProblemType,StateType>* offloadBuffer)
{
    if(threadIdx.x == 0)
    {
        offloadBuffer->doOffload(blockIdx.x);
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
