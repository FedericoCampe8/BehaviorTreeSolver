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
__global__ void doOffloadKernel(OffloadBuffer<ProblemType,StateType>* offloadBuffer, bool onlyRestricted);

void waitOffloadCpu(Vector<std::thread>* cpuThreads, uint64_t* cpuOffloadEndTime);

void waitOffloadGpu(uint64_t* gpuOffloadEndTime);

void waitOffloads(Vector<std::thread>* cpuThreads, uint64_t* cpuOffloadEndTime, uint64_t* gpuOffloadEndTime);

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
    unsigned int const gpuMaxWidth = std::stoi(argv[5]);
    unsigned int const lnsEqPercentage = std::stoi(argv[6]);
    unsigned int const lnsNeqPercentage = std::stoi(argv[7]);
    unsigned int const randomSeed = argc > 8 ? std::stoi(argv[8]) : static_cast<unsigned int>(now());
    assert(lnsEqPercentage + lnsNeqPercentage <= 100);

    // *******************
    // Data initialization
    // *******************
    printf("[INFO] Random seed: %u\n", randomSeed);

    // Parallelism
    unsigned int const cpuWorkloadMultiplier = 4;
    unsigned int const gpuWorkloadMultiplier = 2;
    unsigned int cpuMaxParallelism = 0;
    if (cpuMaxWidth > 0)
    {
        unsigned int const coresCount = std::thread::hardware_concurrency();
        cpuMaxParallelism = cpuWorkloadMultiplier * coresCount;
        printf("[INFO] CPU max parallelism: %u\n", cpuMaxParallelism);
    }
    unsigned int gpuMaxParallelism = 0;
    if (gpuMaxWidth > 0)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        unsigned int const multiProcessorCount = deviceProp.multiProcessorCount;
        unsigned int const maxBlocksPerMultiProcessor = deviceProp.maxBlocksPerMultiProcessor;
        gpuMaxParallelism = gpuWorkloadMultiplier * multiProcessorCount * maxBlocksPerMultiProcessor;
        printf("[INFO] GPU max parallelism: %u\n", gpuMaxParallelism);
    }

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
    printf("[INFO] Start branch and bound search\n");
    uint64_t searchStartTime = now();
    do
    {
        switch(searchStatus)
        {
            case SearchStatus::BB:
            {
                if (priorityQueue.isFull())
                {
                    searchStatus = SearchStatus::LNS;
                    clearLine();
                    printf("[INFO] Switching to large neighborhood search\n");
                }

                prepareOffload(bestSolution, &filteredStatesCount, &priorityQueue, cpuOffloadBuffer);
                prepareOffload(bestSolution, &filteredStatesCount, &priorityQueue, gpuOffloadBuffer);
            }
                break;
            case SearchStatus::LNS:
            {
                prepareOffload(&augmentedRoot, cpuOffloadBuffer);
                prepareOffload(&augmentedRoot, gpuOffloadBuffer);
                cpuOffloadBuffer->generateNeighbourhoods(currentSolution, lnsEqPercentage, lnsNeqPercentage, &rng);
                gpuOffloadBuffer->generateNeighbourhoods(currentSolution, lnsEqPercentage, lnsNeqPercentage, &rng);
                currentSolution->setInvalid();
            }
                break;
        }

        uint64_t cpuOffloadStartTime = now();
        uint64_t gpuOffloadStartTime = now();
        doOffloadsAsync(cpuOffloadBuffer, cpuThreads, gpuOffloadBuffer, searchStatus == SearchStatus::LNS);

        uint64_t cpuOffloadEndTime;
        uint64_t gpuOffloadEndTime;
        waitOffloads(cpuThreads, &cpuOffloadEndTime, &gpuOffloadEndTime);

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
                uint64_t cpuOffloadElapsedTime = max(1ul, cpuOffloadEndTime - cpuOffloadStartTime);
                cpuSpeed = cpuOffloadBuffer->getSize() * 1000 / cpuOffloadElapsedTime;
            }

            unsigned long int gpuSpeed = 0;
            if (gpuOffloadBuffer->getSize() > 0)
            {
                uint64_t gpuOffloadElapsedTime = max(1ul, gpuOffloadEndTime - gpuOffloadStartTime);
                gpuSpeed = gpuOffloadBuffer->getSize() * 1000 / gpuOffloadElapsedTime;
            }
            printf("[INFO] Solution: ");
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

    for (unsigned int index = 0; index < offloadBuffer->getSize(); index += 1)
    {
        StateType const * const approximateSolution = &offloadBuffer->getMDD(index)->bottom;
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
    for (unsigned int index = 0; index < cpuOffloadBuffer->getSize(); index += 1)
    {
        cpuThreads->resize(cpuThreads->getSize() + 1);
        new (cpuThreads->back()) std::thread(&OffloadBuffer<ProblemType,StateType>::doOffload, cpuOffloadBuffer, index, onlyRestricted);
    }
}

template<typename ProblemType, typename StateType>
void doOffloadGpuAsync(OffloadBuffer<ProblemType,StateType>* gpuOffloadBuffer, bool onlyRestricted)
{
    if(not gpuOffloadBuffer->isEmpty())
    {
        DD::MDD<ProblemType,StateType> const * const mdd = gpuOffloadBuffer->getMDD(0);
        unsigned int const blocksCount = gpuOffloadBuffer->getSize();
        unsigned int const blockSize = mdd->width * mdd->problem->maxBranchingFactor;
        assert(blockSize <= 1024);
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

void waitOffloadGpu(uint64_t* gpuOffloadEndTime)
{
    cudaDeviceSynchronize();
    *gpuOffloadEndTime = now();
}

void waitOffloads(Vector<std::thread>* cpuThreads, uint64_t* cpuOffloadEndTime, uint64_t* gpuOffloadEndTime)
{
    std::thread waitCpu(waitOffloadCpu, cpuThreads, cpuOffloadEndTime);
    std::thread waitGpu(waitOffloadGpu, gpuOffloadEndTime);

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

    unsigned int s = ms / 1000;

    printf("%lums (%02uh%02um%02us)", elapsedTimeMs, h, m, s);
}

void clearLine()
{
    // ANSI clear line escape code
    printf("\33[2K\r");
}
