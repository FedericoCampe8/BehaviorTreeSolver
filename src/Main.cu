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
#include <External/Json.hpp>

#include "BB/OffloadQueue.cuh"
#include "BB/MainQueue.cuh"
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
template<typename StateType>
bool hasBiggerCost(QueuedState<StateType> const & queuedState0, QueuedState<StateType> const & queuedState1);

template<typename StateType>
bool hasSmallerCost(QueuedState<StateType> const & queuedState0, QueuedState<StateType> const & queuedState1);

// Queues
bool boundsChk(unsigned int bestCost, unsigned int lowerbound, unsigned int upperbound, unsigned int cost);

template<typename ProblemType, typename StateType, typename ModelType>
void updateMainQueue(unsigned int bestCost, MainQueue<ProblemType,StateType,ModelType>& mainQueue, OffloadQueue<ProblemType,StateType,ModelType>& offloadQueue);

// Search
template<typename ModelType, typename ProblemType, typename StateType>
bool checkForBetterSolutions(StateType& bestSolution, OffloadQueue<ModelType,ProblemType,StateType>& offloadQueue);

// Offload
template<typename ModelType, typename ProblemType, typename StateType>
void prepareOffloadCpu(unsigned int bestCost, MainQueue<ModelType,ProblemType,StateType>& mainQueue, OffloadQueue<ModelType, ProblemType, StateType>& cpuQueue);

template<typename ModelType, typename ProblemType, typename StateType>
void prepareOffloadGpu(unsigned int bestCost, MainQueue<ModelType,ProblemType,StateType>& mainQueue, OffloadQueue<ModelType,ProblemType,StateType>& gpuQueue);

template<typename ModelType, typename ProblemType, typename StateType>
void doOffloadCpu(DD::MDD<ModelType,ProblemType,StateType> const & mdd, OffloadQueue<ModelType,ProblemType,StateType>& cpuQueue, std::byte* scratchpadMem);

template<typename ModelType, typename ProblemType, typename StateType>
void doOffloadGpu(DD::MDD<ModelType,ProblemType,StateType> const & mdd, OffloadQueue<ModelType,ProblemType,StateType>& gpuQueue);

template<typename ModelType, typename ProblemType, typename StateType>
__host__ __device__ void doOffload(DD::MDD<ModelType,ProblemType,StateType> const & mdd, BB::OffloadedState<StateType>& offloadedState, std::byte* scratchpadMem);

template<typename ModelType, typename ProblemType, typename StateType>
__global__ void doOffloadKernel(DD::MDD<ModelType,ProblemType,StateType> const & mdd, Vector<BB::OffloadedState<StateType>>& queue);

// Debug
void printElapsedTime(uint64_t elapsedTimeMs);

int main(int argc, char ** argv)
{
    // Input parsing
    char const * problemFileName = argv[1];
    unsigned int const mddMaxWidth = std::stoi(argv[2]);
    unsigned int const timeoutSeconds = std::stoi(argv[3]);
    unsigned int const queueMaxSize = std::stoi(argv[4]);
    unsigned int const cpuMaxParallelism = std::stoi(argv[5]);
    unsigned int const gpuMaxParallelism = std::stoi(argv[6]);

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
    ModelType * const model = reinterpret_cast<ModelType*>(memory);
    new (model) ModelType(*problem);

    // MDD
    memorySize = sizeof(MDD<ModelType,ProblemType,StateType>);
    memory = safeMalloc(memorySize, gpuDataMallocType);
    MDD<ModelType,ProblemType,StateType>* const mdd = reinterpret_cast<MDD<ModelType,ProblemType,StateType>*>(memory);
    new (mdd) MDD<ModelType,ProblemType,StateType>(*model, mddMaxWidth);

    // Context initialization
    std::byte* scratchpadMem = nullptr;
    if(cpuMaxParallelism > 0)
    {
        scratchpadMem = safeMalloc(mdd->scratchpadMemSize * cpuMaxParallelism, MallocType::Std);
    }

    // Main queue
    MainQueue<ModelType,ProblemType,StateType> mainQueue(*mdd, queueMaxSize, hasSmallerCost<StateType>, hasBiggerCost<StateType>);

    // Offload queues
    OffloadQueue<ModelType,ProblemType,StateType> cpuQueue(*mdd, cpuMaxParallelism, MallocType::Std);
    OffloadQueue<ModelType,ProblemType,StateType> gpuQueue(*mdd, gpuMaxParallelism, gpuDataMallocType);

    // Best solution
    unsigned int const stateSize = sizeof(VRPState);
    memory = safeMalloc(stateSize, gpuDataMallocType);
    StateType* bestSolution = reinterpret_cast<StateType*>(memory);
    memory = StateType::mallocStorages(1, *problem, MallocType::Std);
    new (bestSolution) StateType(*problem, memory);
    bestSolution->cost = StateType::MaxCost;

    // Root
    memory = safeMalloc(stateSize, gpuDataMallocType);
    VRPState* root = reinterpret_cast<StateType*>(memory);
    memory = StateType::mallocStorages(1, *problem, gpuDataMallocType);
    new (root) StateType(*problem, memory);
    mdd->model.makeRoot(*root);

    // Enqueue root
    mainQueue.enqueue(VRPState::MaxCost, VRPState::MaxCost, *root);

    // Search
    unsigned int visitedStatesCount = 0;
    unsigned int iterationsCount = 0;
    uint64_t searchStartTime = now();
    do
    {
        prepareOffloadCpu<ModelType>(bestSolution->cost, mainQueue, cpuQueue);
        uint64_t cpuOffloadStartTime = now();
        if (not cpuQueue.queue.isEmpty())
        {
            doOffloadCpu<ModelType>(*mdd, cpuQueue, scratchpadMem);
            visitedStatesCount += cpuQueue.queue.getSize();
        }

        prepareOffloadGpu<ModelType>(bestSolution->cost, mainQueue, gpuQueue);
        uint64_t gpuOffloadStartTime = now();
        if (not gpuQueue.queue.isEmpty())
        {
            doOffloadGpu<ModelType>(*mdd, gpuQueue);
            visitedStatesCount += gpuQueue.queue.getSize();
        }

        bool foundBetterSolution =
            checkForBetterSolutions<ModelType>(*bestSolution, cpuQueue) or
            checkForBetterSolutions<ModelType>(*bestSolution, gpuQueue);

        if(foundBetterSolution)
        {
            printf("[INFO] Better solution found: ");
            bestSolution->selectedValues.print(false);
            printf(" | Value: %u", bestSolution->cost);
            printf(" | Time: ");
            printElapsedTime(Chrono::now() - searchStartTime);
            printf(" | Iterations: %u", iterationsCount);
            printf(" | Visited states: %u\n", visitedStatesCount);
        }
        else
        {
            printf("[INFO] CPU Speed: %5lu states/s", static_cast<uint64_t>(cpuQueue.queue.getSize()) * 1000 / (Chrono::now() - cpuOffloadStartTime));
            printf(" | GPU Speed: %5lu states/s", static_cast<uint64_t>(gpuQueue.queue.getSize()) * 1000 / (Chrono::now() - gpuOffloadStartTime));
            printf(" | Time: ");
            printElapsedTime(Chrono::now() - searchStartTime);
            printf(" | Iterations: %u", iterationsCount);
            printf(" | State to visit: %u", mainQueue.getSize());
            printf(" | Visited states: %u\r", visitedStatesCount);
        }

        updateMainQueue<ModelType,ProblemType,StateType>(bestSolution->cost, mainQueue, cpuQueue);
        updateMainQueue<ModelType,ProblemType,StateType>(bestSolution->cost, mainQueue, gpuQueue);

        iterationsCount += 1;
    }
    while((not mainQueue.isEmpty()) and ((Chrono::now() - searchStartTime) < timeoutSeconds * 1000));

    printf("[RESULT] Solution: ");
    bestSolution->selectedValues.print(false);
    printf(" | Value: %u", bestSolution->cost);
    printf(" | Time: ");
    printElapsedTime(Chrono::now() - searchStartTime);
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

    // Malloc problem
    unsigned int const problemSize = sizeof(sizeof(OP::VRProblem));
    std::byte* const memory = Memory::safeMalloc(problemSize, mallocType);
    OP::VRProblem* const problem = reinterpret_cast<OP::VRProblem*>(memory);

    // Init problem
    unsigned int const variablesCount = problemJson["nodes"].size();
    new (problem) OP::VRProblem(variablesCount, mallocType);

    // Init variables
    new (&problem->variables[0]) OP::Variable(0, 0);
    thrust::for_each(problem->variables.begin(), problem->variables.end(), [&] (OP::Variable& variable)
    {
        new (&variable) OP::Variable(2, variablesCount - 1);
    });
    new (&problem->variables[variablesCount - 1]) OP::Variable(1, 1);

    // Init start/end locations
    problem->start = 0;
    problem->end = 1;

    // Init pickups and deliveries
    for(unsigned int i = 2; i < variablesCount; i += 2)
    {
        problem->pickups.pushBack(i);
        problem->deliveries.pushBack(i + 1);
    }

    // Init distances
    for(unsigned int from = 0; from < variablesCount; from += 1)
    {
        for(unsigned int to = 0; to < variablesCount; to += 1)
        {
            problem->distances[(from * variablesCount) + to] = problemJson["edges"][from][to];
        }
    }

    return problem;
}

template<typename StateType>
bool hasBiggerCost(QueuedState<StateType> const & queuedState0, QueuedState<StateType> const & queuedState1)
{
    return queuedState0.state.cost > queuedState1.state.cost;
}

template<typename StateType>
bool hasSmallerCost(QueuedState<StateType> const & queuedState0, QueuedState<StateType> const & queuedState1)
{
    return queuedState0.state.cost < queuedState1.state.cost;
}

template<typename ModelType, typename ProblemType, typename StateType>
void updateMainQueue(unsigned int bestCost, MainQueue<ModelType,ProblemType,StateType>& mainQueue, OffloadQueue<ModelType,ProblemType,StateType>& offloadQueue)
{
    assert(mainQueue.getSize() + (offloadQueue.cutsetMaxSize * offloadQueue.queue.getSize()) < mainQueue.getCapacity());

    thrust::for_each(thrust::seq, offloadQueue.queue.begin(), offloadQueue.queue.end(), [&] (OffloadedState<StateType>& offloadedState)
    {
        if(boundsChk(bestCost, offloadedState.lowerbound, offloadedState.upperbound, offloadedState.state.cost))
        {
            thrust::for_each(thrust::seq, offloadedState.cutset.begin(), offloadedState.cutset.end(), [&] (StateType& cutsetState)
            {
                mainQueue.enqueue(offloadedState.lowerbound, offloadedState.upperbound, cutsetState);
            });
        }
    });
}

bool boundsChk(unsigned int bestCost, unsigned int lowerbound, unsigned int upperbound, unsigned int cost)
{
    return
        lowerbound < upperbound and
        lowerbound < bestCost and
        cost < bestCost;
}

template<typename ModelType, typename ProblemType, typename StateType>
bool checkForBetterSolutions(StateType& bestSolution, OffloadQueue<ModelType,ProblemType,StateType>& offloadQueue)
{
    bool foundBetterSolution = false;

    thrust::for_each(offloadQueue.queue.begin(), offloadQueue.queue.end(), [&] (OffloadedState<StateType>& offloadedState)
    {
        if (offloadedState.upperbound < bestSolution.cost)
        {
            bestSolution = offloadedState.upperboundState;
            foundBetterSolution = true;
        }
    });

    return foundBetterSolution;
}

template<typename ModelType, typename ProblemType, typename StateType>
void prepareOffloadCpu(unsigned int bestCost, MainQueue<ModelType,ProblemType,StateType>& mainQueue, OffloadQueue<ModelType, ProblemType, StateType>& cpuQueue)
{
    cpuQueue.queue.clear();
    while(not (mainQueue.isEmpty() or cpuQueue.queue.isFull()))
    {
        QueuedState<StateType> const & queuedState = mainQueue.getStateForCpu();
        if(boundsChk(bestCost, queuedState.lowerbound, queuedState.upperbound, queuedState.state.cost))
        {
            cpuQueue.enqueue(queuedState.state);
        }
        mainQueue.dequeue(queuedState);
    }
}

template<typename ModelType, typename ProblemType, typename StateType>
void prepareOffloadGpu(unsigned int bestCost, MainQueue<ModelType,ProblemType,StateType>& mainQueue, OffloadQueue<ModelType,ProblemType,StateType>& gpuQueue)
{
    gpuQueue.queue.clear();
    while(not (mainQueue.isEmpty() or gpuQueue.queue.isFull()))
    {
        QueuedState<StateType> const & queuedState = mainQueue.getStateForGpu();
        if(boundsChk(bestCost, queuedState.lowerbound, queuedState.upperbound, queuedState.state.cost))
        {
            gpuQueue.enqueue(queuedState.state);
        }
        mainQueue.dequeue(queuedState);
    }
}

template<typename ModelType, typename ProblemType, typename StateType>
void doOffloadCpu(DD::MDD<ModelType,ProblemType,StateType> const & mdd, OffloadQueue<ModelType,ProblemType,StateType>& cpuQueue, std::byte* scratchpadMem)
{
    thrust::for_each(thrust::host, cpuQueue.queue.begin(), cpuQueue.queue.end(), [&] (OffloadedState<StateType>& offloadedState)
    {
        unsigned int stateIdx = thrust::distance(cpuQueue.queue.begin(), &offloadedState);
        doOffload(mdd, offloadedState, &scratchpadMem[mdd.scratchpadMemSize * stateIdx]);
    });
}

template<typename ModelType, typename ProblemType, typename StateType>
void doOffloadGpu(DD::MDD<ModelType,ProblemType,StateType> const & mdd, OffloadQueue<ModelType,ProblemType,StateType>& gpuQueue)
{
    doOffloadKernel<ModelType,ProblemType,StateType><<<gpuQueue.queue.getSize(), 1, mdd.scratchpadMemSize>>>(mdd, gpuQueue.queue);
    cudaDeviceSynchronize();
}

template<typename ModelType, typename ProblemType, typename StateType>
__host__ __device__
void doOffload(DD::MDD<ModelType,ProblemType,StateType> const & mdd, BB::OffloadedState<StateType>& offloadedState, std::byte* scratchpadMem)
{
    //Preparation to build MDDs
    StateType const & top = offloadedState.state;
    Vector<StateType>& cutset = offloadedState.cutset;
    StateType& bottom = offloadedState.upperboundState;

    //Build MDDs
    mdd.buildTopDown(DD::MDD<ModelType,ProblemType,StateType>::Type::Relaxed, top, cutset, bottom, scratchpadMem);
    offloadedState.lowerbound = bottom.cost;
    mdd.buildTopDown(DD::MDD<ModelType,ProblemType,StateType>::Type::Restricted, top, cutset, bottom, scratchpadMem);
    offloadedState.upperbound = bottom.cost;
}

template<typename ModelType, typename ProblemType, typename StateType>
__global__
void doOffloadKernel(DD::MDD<ModelType,ProblemType,StateType> const & mdd, Vector<BB::OffloadedState<StateType>>& queue)
{
    extern __shared__ unsigned int sharedMem[];
    std::byte* scratchpadMem = reinterpret_cast<std::byte*>(sharedMem);

    if(blockIdx.x * blockDim.x + threadIdx.x == 0)
    {
        BB::OffloadedState<StateType>& offloadedState = queue.at(blockIdx.x);
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