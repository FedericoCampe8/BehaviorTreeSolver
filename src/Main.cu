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

using namespace BB;
using namespace Chrono;
using namespace DD;
using namespace DP;
using namespace Memory;
using namespace OP;

using ModelType = VRPModel;
using ProblemType = ModelType::ProblemType;
using StateType = ModelType::StateType;

// Auxiliary functions
void configGPU();
OP::VRProblem* parseGrubHubInstance(char const * problemFileName, Memory::MallocType mallocType);

// Queues
template<typename T>
bool dummyCmp(QueuedState<T> const & queuedState0, QueuedState<T> const & queuedState1);

template<typename T>
void updateMainQueue(unsigned int bestCost, MainQueue<T>& mainQueue, OffloadQueue<T>& offloadQueue);

// Search
template<typename T>
bool boundsChk(unsigned int bestCost, unsigned int lowerbound, unsigned int upperbound, unsigned int cost);

template<typename T>
bool boundsChk(unsigned int bestCost, QueuedState<T> const & queuedState);

template<typename T>
bool boundsChk(unsigned int bestCost, OffloadedState<T> const & offloadedState);

template<typename T>
bool checkForBetterSolutions(T::StateType& bestSolution, OffloadQueue<T>& offloadQueue);

// Offload
template<typename T>
void prepareOffloadCpu(unsigned int bestCost, MainQueue<T>& queueManager, OffloadQueue<T>& cpuQueue);

template<typename T>
void prepareOffloadGpu(unsigned int bestCost, MainQueue<T>& mainQueue, OffloadQueue<T>& gpuQueue);

template<typename T>
void doOffloadCpu(DD::MDD<T> const & mdd, OffloadQueue<T>& cpuQueue, std::byte* scratchpadMem);

template<typename T>
void doOffloadGpu(DD::MDD<T> const & mdd, OffloadQueue<T>& gpuQueue);

template<typename T>
__host__ __device__ void doOffload(DD::MDD<T> const & mdd, BB::OffloadedState<T::StateType>& offloadedState, std::byte* scratchpadMem);

template<typename T>
__global__ void doOffloadKernel(DD::MDD<T> const & mdd, Vector<BB::OffloadedState<T::StateType>>& queue);

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
    memorySize = sizeof(MDD<ModelType>);
    memory = safeMalloc(memorySize, gpuDataMallocType);
    MDD<ModelType>* const mdd = reinterpret_cast<MDD<ModelType>*>(memory);
    new (mdd) MDD<ModelType>(*model, mddMaxWidth);

    // Context initialization
    std::byte* scratchpadMem = nullptr;
    if(cpuMaxParallelism > 0)
    {
        scratchpadMem = safeMalloc(mdd->scratchpadMemSize * cpuMaxParallelism, MallocType::Std);
    }

    // Main queue
    MainQueue<ModelType> mainQueue(*mdd, queueMaxSize, dummyCmp<QueuedState<QueuedState>>, dummyCmp<QueuedState<QueuedState>>);

    // Offload queues
    OffloadQueue<ModelType> cpuQueue(*mdd, cpuMaxParallelism, MallocType::Std);
    OffloadQueue<ModelType> gpuQueue(*mdd, gpuMaxParallelism, gpuDataMallocType);

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

        updateMainQueue<ModelType>(bestSolution->cost, mainQueue, cpuQueue);
        updateMainQueue<ModelType>(bestSolution->cost, mainQueue, cpuQueue);

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

template<typename T>
bool dummyCmp(QueuedState<T> const & queuedState0, QueuedState<T> const & queuedState1s)
{
    return false;
}

template<typename T>
void updateMainQueue(unsigned int bestCost, MainQueue<T>& mainQueue, OffloadQueue<T>& offloadQueue)
{
    assert(mainQueue.getSize() + (offloadQueue.cutsetMaxSize * offloadQueue.queue.getSize()) < mainQueue.getCapacity());

    thrust::for_each(thrust::seq, offloadQueue.queue.begin(), offloadQueue.queue.end(), [&] (OffloadedState<T::StateType>& offloadedState)
    {
        if(boundsChk(bestCost, offloadedState))
        {
            thrust::for_each(thrust::seq, offloadedState.cutset.begin(), offloadedState.cutset.end(), [&] (T::StateType& cutsetState)
            {
                mainQueue.enqueue(offloadedState.lowerbound, offloadedState.upperbound, cutsetState);
            });
        }
    });
}

template<typename T>
bool boundsChk(unsigned int bestCost, unsigned int lowerbound, unsigned int upperbound, unsigned int cost)
{
    return
        lowerbound < upperbound and
        lowerbound < bestCost and
        cost < bestCost;
}

template<typename T>
bool boundsChk(unsigned int bestCost, QueuedState<T> const & queuedState)
{
    return boundsChk(bestCost, queuedState.lowerbound, queuedState.upperbound, queuedState.state.cost);
}

template<typename T>
bool boundsChk(unsigned int bestCost, OffloadedState<T> const & offloadedState)
{
    return boundsChk(bestCost, offloadedState.lowerbound, offloadedState.upperbound, offloadedState.state.cost);
}

template<typename T>
bool checkForBetterSolutions(T::StateType& bestSolution, OffloadQueue<T>& offloadQueue)
{
    bool foundBetterSolution = false;

    thrust::for_each(offloadQueue.queue.begin(), offloadQueue.queue.end(), [&] (OffloadedState<T::StateType>& offloadedState)
    {
        if (offloadedState.upperbound < bestSolution.cost)
        {
            bestSolution = offloadedState.upperboundState;
            foundBetterSolution = true;
        }
    });

    return foundBetterSolution;
}

template<typename T>
void prepareOffloadCpu(unsigned int bestCost, MainQueue<T>& mainQueue, OffloadQueue<T>& cpuQueue)
{
    cpuQueue.queue.clear();
    while(not (mainQueue.isEmpty() or cpuQueue.queue.isFull()))
    {
        QueuedState<T::StateType> const & queuedState = mainQueue.getStateForCpu();
        if(boundsChk<T::StateType>(bestCost, queuedState))
        {
            cpuQueue.enqueue(queuedState.state);
        }
        mainQueue.dequeue(queuedState);
    }
}

template<typename T>
void prepareOffloadGpu(unsigned int bestCost, MainQueue<T>& mainQueue, OffloadQueue<T>& gpuQueue)
{
    gpuQueue.queue.clear();
    while(not (mainQueue.isEmpty() or gpuQueue.queue.isFull()))
    {
        QueuedState<T::StateType> const & queuedState = mainQueue.getStateForGpu();
        if(boundsChk<T::StateType>(bestCost, queuedState))
        {
            gpuQueue.enqueue(queuedState.state);
        }
        mainQueue.dequeue(queuedState);
    }
}

template<typename T>
void doOffloadCpu(DD::MDD<T> const & mdd, OffloadQueue<T>& cpuQueue, std::byte* scratchpadMem)
{
    thrust::for_each(thrust::host, cpuQueue.queue.begin(), cpuQueue.queue.end(), [&] (OffloadedState<T::StateTypr>& offloadedState)
    {
        unsigned int stateIdx = thrust::distance(cpuQueue.queue.begin(), &offloadedState);
        doOffload(mdd, offloadedState, &scratchpadMem[mdd.scratchpadMemSize * stateIdx]);
    });
}

template<typename T>
void doOffloadGpu(DD::MDD<T> const & mdd, OffloadQueue<T>& gpuQueue)
{
    doOffloadKernel<<<gpuQueue.queue.getSize(), 1, mdd.scratchpadMemSize>>>(mdd, gpuQueue.queue);
    cudaDeviceSynchronize();
}

template<typename T>
__host__ __device__
void doOffload(DD::MDD<T> const & mdd, BB::OffloadedState<T::StateType>& offloadedState, std::byte* scratchpadMem)
{
    //Preparation to build MDDs
    T::StateType& top = offloadedState.state;
    Vector<T::StateType>& cutset = offloadedState.cutset;
    T::StateType& bottom = offloadedState.upperboundState;

    //Build MDDs
    mdd.buildTopDown(DD::MDD<T>::Type::Relaxed, top, cutset, bottom, scratchpadMem);
    offloadedState.lowerbound = bottom.cost;
    mdd.buildTopDown(DD::MDD<T>::Type::Restricted, top, cutset, bottom, scratchpadMem);
    offloadedState.upperbound = bottom.cost;
}

template<typename T>
__global__
void doOffloadKernel(DD::MDD<T> const & mdd, Vector<BB::OffloadedState<T::StateType>>& queue)
{
    extern __shared__ unsigned int sharedMem[];
    std::byte* scratchpadMem = reinterpret_cast<std::byte*>(sharedMem);

    if(blockIdx.x * blockDim.x + threadIdx.x == 0)
    {
        BB::OffloadedState<T::StateType>& offloadedState = queue.at(blockIdx.x);
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