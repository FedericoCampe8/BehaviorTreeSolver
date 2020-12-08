#include <cstdio>
#include <cinttypes>
#include <cstddef>
#include <new>
#include <utility>
#include <algorithm>
#include <fstream>

#include <Utils/Memory.cuh>
#include <Utils/Chrono.cuh>
#include <Utils/CUDA.cuh>
#include <Containers/ManualVector.cuh>
#include <External/Json.hpp>

#include "OP/TSPProblem.cuh"
#include "MDD/MDD.cuh"
#include "DP/TSPModel.cuh"
#include "BB/FullState.cuh"

#define CPU_QUEUE_SIZE 10000000

using namespace std;
using json = nlohmann::json;

//Auxiliary functions
void setupGPU();

OP::TSPProblem* parseGrubHubInstance(char const * problemFileName);

ManualVector<BB::FullState<DP::TSPState>>* fullStateBuffer(OP::TSPProblem const * problem, unsigned int size);
RuntimeArray<DP::TSPState>* stateArray(OP::TSPProblem const * problem, unsigned int size);
ManualVector<unsigned int>* uintManualVector(unsigned int size);
RuntimeArray<unsigned int>* uintArray(unsigned int size);

unsigned int enqueueStatesToGPU(unsigned int globalUpperBound, ManualVector<BB::FullState<DP::TSPState>>* queue, ManualVector<unsigned int>* toOffload);
void doOffload(OP::TSPProblem const * problem, unsigned int width, ManualVector<unsigned int>* toOffload, ManualVector<BB::FullState<DP::TSPState>>* queue, RuntimeArray<DP::TSPState>* solutions, RuntimeArray<unsigned int>* cutsetsSizes, ManualVector<BB::FullState<DP::TSPState>>* cutsets);
__global__ void offload(OP::TSPProblem const * problem, unsigned int width, ManualVector<unsigned int>* toOffload, ManualVector<BB::FullState<DP::TSPState>>* queue, RuntimeArray<DP::TSPState>* solutions, RuntimeArray<unsigned int>* cutsetsSizes, ManualVector<BB::FullState<DP::TSPState>>* cutsets);
void checkForBestSolutions(uint64_t startTime, uint64_t& bestSolutionTime, DP::TSPState& bestSolution, ManualVector<unsigned int>* toOffload, ManualVector<BB::FullState<DP::TSPState>>* queue, RuntimeArray<DP::TSPState>* solutions);
unsigned int dequeueStatesFromGPU(unsigned int globalUpperBound, ManualVector<unsigned int>* toOffload, unsigned int cutsetMaxSize, RuntimeArray<unsigned int>* cutsetsSizes, ManualVector<BB::FullState<DP::TSPState>>* cutsets, ManualVector<BB::FullState<DP::TSPState>>* queue);

int main(int argc, char ** argv)
{
    char const * problemFileName = argv[1];
    unsigned int width = std::atoi(argv[2]);
    unsigned int maxParallelism = std::atoi(argv[3]);

    setupGPU();

    auto* const problem = parseGrubHubInstance(problemFileName);

    // Queue
    ManualVector<BB::FullState<DP::TSPState>>* const queue = fullStateBuffer(problem, CPU_QUEUE_SIZE);

    // Cutsets
    unsigned int cutsetMaxSize = width * MDD::MDD::calcFanout(problem);
    ManualVector<BB::FullState<DP::TSPState>>* const cutsets = fullStateBuffer(problem, cutsetMaxSize * maxParallelism);
    RuntimeArray<unsigned int>* const cutsetsSizes = uintArray(maxParallelism);

    // To offload
    ManualVector<unsigned int>* const toOffload = uintManualVector(maxParallelism + 1);

    // Root
    DP::TSPModel::makeRoot(problem, queue->at(0).state);
    queue->emplaceBackValid(true, queue->at(0).state);

    //Solutions
    RuntimeArray<DP::TSPState>* solutions = stateArray(problem, maxParallelism + 1);

    //Optimum
    DP::TSPState& bestSolution = solutions->at(solutions->size - 1);
    DP::TSPState::reset(bestSolution);
    uint64_t bestSolutionTime = Chrono::now();

    unsigned int visitedStates = 0;
    auto startTime = Chrono::now();

    unsigned int enqueuedStates;
    unsigned int iterationsCount = 0;


    do
    {
        /*
        printf("[INFO] --- Iteration %d ---\n", iterationsCount);
        printf("[INFO] Best solution: ");
        bestSolution.selectedValues.print(false);
        printf(" | Value: %d | Time %ld ms\n", bestSolution.cost, bestSolutionTime > startTime ? static_cast<long>(Chrono::now() - startTime) : 0);
        printf("[INFO] Visited states: %u\n", visitedStates);
        printf("[INFO] Total time: %ld ms\n", static_cast<long>(Chrono::now() - startTime));
         */

        enqueuedStates = enqueueStatesToGPU(bestSolution.cost, queue, toOffload);

        if (enqueuedStates > 0)
        {
            visitedStates += toOffload->validPrefixSize;

            doOffload(problem, width, toOffload, queue, solutions, cutsetsSizes, cutsets);

            checkForBestSolutions(startTime, bestSolutionTime, bestSolution, toOffload, queue, solutions);

            dequeueStatesFromGPU(bestSolution.cost, toOffload, cutsetMaxSize, cutsetsSizes, cutsets, queue);

            iterationsCount += 1;
        }
    }
    while(enqueuedStates > 0);

    return EXIT_SUCCESS;
}

void setupGPU()
{
    // Heap
    std::size_t sizeHeap = 3ul * 1024ul * 1024ul * 1024ul;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, sizeHeap);

    // Stack
    size_t sizeStackThread = 4 * 1024;
    cudaDeviceSetLimit(cudaLimitStackSize, sizeStackThread);

    // Cache
    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
}

OP::TSPProblem * parseGrubHubInstance(char const * problemFileName)
{
    // Parse
    std::ifstream problemFile(problemFileName);
    json problemJson;
    problemFile >> problemJson;

    // Alloc
    unsigned int varsCount = problemJson["nodes"].size();
    std::size_t problemSize = sizeof(OP::TSPProblem);
    std::size_t problemStorageSize = OP::TSPProblem::sizeofStorage(varsCount);
    std::byte* mem = Memory::safeManagedMalloc(problemSize + problemStorageSize);
    OP::TSPProblem* problem = reinterpret_cast<OP::TSPProblem*>(mem);
    byte* problemStorage = mem + problemSize;

    new (problem) OP::TSPProblem(varsCount, problemStorage);

    // Variables
    new (&problem->vars[0]) OP::Variable(0,0);
    for(unsigned int varIdx = 1; varIdx < varsCount - 1; varIdx += 1)
    {
        new (&problem->vars[varIdx]) OP::Variable(2,varsCount - 1);
    }
    new (&problem->vars[varsCount - 1]) OP::Variable(1,1);

    problem->setStartEndLocations(0,1);

    // Precedences
    for(unsigned int i = 2; i < varsCount; i += 2)
    {
        problem->addPickupDelivery(i,i + 1);
    }

    // Distances
    for(unsigned int i = 0; i < varsCount; i += 1)
    {
        for(unsigned int j = 0; j < varsCount; j += 1)
        {
            problem->distances[(i * varsCount) + j] = problemJson["edges"][i][j];
        }
    }

    return problem;
}

ManualVector<BB::FullState<DP::TSPState>>* fullStateBuffer(OP::TSPProblem const * problem, unsigned int size)
{
    assert(size != 0);

    // Malloc
    RuntimeArray<DP::TSPState>* states = stateArray(problem, size);

    std::byte* fullStatesMem = Memory::safeManagedMalloc(sizeof(BB::FullState<DP::TSPState>) * size);
    BB::FullState<DP::TSPState>* fullStates = reinterpret_cast<BB::FullState<DP::TSPState>*>(fullStatesMem);

    std::byte* bufferMem = Memory::safeManagedMalloc(sizeof(ManualVector<BB::FullState<DP::TSPState>>));
    ManualVector<BB::FullState<DP::TSPState>>* buffer = reinterpret_cast<ManualVector<BB::FullState<DP::TSPState>>*>(bufferMem);

    // Init
    for(unsigned int fullStateIdx = 0; fullStateIdx < size; fullStateIdx +=1)
    {
        new (&fullStates[fullStateIdx]) BB::FullState<DP::TSPState>(false, &states->at(fullStateIdx));
    };

    new (buffer) ManualVector<BB::FullState<DP::TSPState>>(size, fullStatesMem);

    return buffer;
}

RuntimeArray<DP::TSPState>* stateArray(OP::TSPProblem const * problem, unsigned int size)
{
    assert(size != 0);

    // Malloc
    std::byte* statesMem = Memory::safeManagedMalloc(sizeof(DP::TSPState) * size);
    DP::TSPState* states = reinterpret_cast<DP::TSPState*>(statesMem);

    std::size_t stateStorageSize = DP::TSPState::sizeofStorage(problem);
    std::byte* statesStorage = Memory::safeManagedMalloc(stateStorageSize * size);

    std::byte* arrayMem = Memory::safeManagedMalloc(sizeof(RuntimeArray<DP::TSPState>));
    RuntimeArray<DP::TSPState>* array = reinterpret_cast<RuntimeArray<DP::TSPState>*>(arrayMem);

    // Init
    for(unsigned int stateIdx = 0; stateIdx < size; stateIdx +=1)
    {
        new (&states[stateIdx]) DP::TSPState(problem, &statesStorage[stateStorageSize * stateIdx]);
    };

    new (array) RuntimeArray<DP::TSPState>(size, statesMem);

    return array;
}

ManualVector<unsigned int>* uintManualVector(unsigned int size)
{
    // Malloc
    std::byte* storage = Memory::safeManagedMalloc(StaticVector<unsigned int>::sizeofStorage(size));

    std::byte* vectorMem = Memory::safeManagedMalloc(sizeof(ManualVector<unsigned int>));
    ManualVector<unsigned int>* vector = reinterpret_cast<ManualVector<unsigned int>*>(vectorMem);

    // Init
    new (vector) ManualVector<unsigned int>(size, storage);

    return vector;
}

RuntimeArray<unsigned int>* uintArray(unsigned int size)
{
    // Malloc
    std::byte* storage = Memory::safeManagedMalloc(RuntimeArray<unsigned int>::sizeofStorage(size));

    std::byte* arrayMem = Memory::safeManagedMalloc(sizeof(RuntimeArray<unsigned int>));
    RuntimeArray<unsigned int>* array = reinterpret_cast<RuntimeArray<unsigned int>*>(arrayMem);

    // Init
    new (array) RuntimeArray<unsigned int>(size, storage);

    return array;
}


unsigned int enqueueStatesToGPU(unsigned int globalUpperBound, ManualVector<BB::FullState<DP::TSPState>>* queue, ManualVector<unsigned int>* toOffload)
{
    assert(toOffload->isEmptyValid());

    unsigned int checkedSuffixSize = 0;
    unsigned int checkedStatesCount = 0;
    while(checkedSuffixSize < queue->validPrefixSize)
    {
        auto cmp = [=] (unsigned int& idx1, unsigned int& idx2)
        {

            DP::TSPState const * state1 = queue->at(idx1).state;
            DP::TSPState const * state2 = queue->at(idx2).state;

            double w1 = static_cast<double>(state1->cost) / static_cast<double>(state1->selectedValues.getSize());
            double w2 = static_cast<double>(state2->cost) / static_cast<double>(state2->selectedValues.getSize());

            return w1 < w2;
        };

        unsigned int fullStateIdx = queue->validPrefixSize - checkedSuffixSize - 1;
        BB::FullState<DP::TSPState>& fullState = queue->at(fullStateIdx);

        if (fullState.active)
        {
            if(fullState.lowerBound < globalUpperBound and fullState.state->cost < globalUpperBound)
            {
                checkedStatesCount += 1;

                toOffload->pushBackValid(fullStateIdx);
                fullState.active = false;
                std::push_heap(toOffload->beginValid(), toOffload->endValid(), cmp);

                if (toOffload->isFullValid())
                {
                    std::pop_heap(toOffload->beginValid(), toOffload->endValid(), cmp);
                    queue->at(toOffload->lastValid()).active = true;
                    toOffload->popBackValid();
                }
            }
            else
            {
                fullState.active = false;
            }
        }

        checkedSuffixSize += 1;
    }

    printf("[INFO] Enqueued %u of %u (%u) states\r", toOffload->validPrefixSize, checkedStatesCount, queue->validPrefixSize);

    return toOffload->validPrefixSize;
}

void doOffload(OP::TSPProblem const * problem, unsigned int width, ManualVector<unsigned int>* toOffload, ManualVector<BB::FullState<DP::TSPState>>* queue, RuntimeArray<DP::TSPState>* solutions, RuntimeArray<unsigned int>* cutsetsSizes, ManualVector<BB::FullState<DP::TSPState>>* cutsets)
{
        offload<<<toOffload->validPrefixSize, 32>>>(problem, width, toOffload, queue, solutions, cutsetsSizes, cutsets);
        cudaDeviceSynchronize();
}

__global__
void offload(OP::TSPProblem const * problem, unsigned int width, ManualVector<unsigned int>* toOffload, ManualVector<BB::FullState<DP::TSPState>>* queue, RuntimeArray<DP::TSPState>* solutions, RuntimeArray<unsigned int>* cutsetsSizes, ManualVector<BB::FullState<DP::TSPState>>* cutsets)
{
    __shared__ unsigned int alignedSharedMem[1000];
    std::byte* sharedMem = reinterpret_cast<std::byte*>(alignedSharedMem);

    ONE_THREAD_IN_BLOCK
    {
        //MDD
        unsigned int fullStateIdx = toOffload->at(blockIdx.x);
        BB::FullState<DP::TSPState>& enqueuedState = queue->at(fullStateIdx);
        MDD::MDD mdd(MDD::MDD::Type::Relaxed, width, enqueuedState.state, problem);

        DP::TSPState* solution = &solutions->at(blockIdx.x);
        unsigned int cutsetMaxSize = width * MDD::MDD::calcFanout(problem);
        unsigned int& cutsetSize = cutsetsSizes->at(blockIdx.x);

        BB::FullState<DP::TSPState>* cutset = &cutsets->at(cutsetMaxSize * blockIdx.x);
        mdd.buildTopDown(solution, cutsetSize, cutset->state, sharedMem);
        enqueuedState.lowerBound = solution->cost;

        mdd.type = MDD::MDD::Type::Restricted;
        mdd.buildTopDown(solution, cutsetSize, cutset->state, sharedMem);
        enqueuedState.upperBound = solution->cost;

        for(unsigned int cutsetStateIdx = 0; cutsetStateIdx < cutsetSize; cutsetStateIdx += 1)
        {
            cutset[cutsetStateIdx].active = true;
        }
    }
}

void checkForBestSolutions(uint64_t startTime, uint64_t& bestSolutionTime, DP::TSPState& bestSolution, ManualVector<unsigned int>* toOffload, ManualVector<BB::FullState<DP::TSPState>>* queue, RuntimeArray<DP::TSPState>* solutions)
{
    for(unsigned int i = 0; i < toOffload->validPrefixSize; i += 1)
    {
        unsigned int fullStateIdx = toOffload->at(i);
        BB::FullState<DP::TSPState>& fullState = queue->at(fullStateIdx);
        if (fullState.upperBound < bestSolution.cost)
        {
            bestSolution = solutions->at(i);
            bestSolutionTime = Chrono::now();
            printf("[INFO] New best solution: ");
            bestSolution.selectedValues.print(false);
            printf(" | Value: %d | Time %ld ms\n", bestSolution.cost, bestSolutionTime > startTime ? static_cast<long>(Chrono::now() - startTime) : 0);

        }
    }
}

unsigned int dequeueStatesFromGPU(unsigned int globalUpperBound, ManualVector<unsigned int>* toOffload, unsigned int cutsetMaxSize, RuntimeArray<unsigned int>* cutsetsSizes, ManualVector<BB::FullState<DP::TSPState>>* cutsets, ManualVector<BB::FullState<DP::TSPState>>* queue)
{
    unsigned int checkedStatesCount = 0;
    unsigned int dequeuedStatesCount = 0;
    unsigned int activePrefixSize = 0;

    for (unsigned int i = 0; i < toOffload->validPrefixSize; i += 1)
    {
        unsigned int enqueuedStateIdx = toOffload->at(i);
        BB::FullState<DP::TSPState>& enqueuedState = queue->at(enqueuedStateIdx);
        unsigned int cutsetSize = cutsetsSizes->at(i);

        if(not enqueuedState.state->selectedValues.isFull())
        {
            for (unsigned int cutsetStateIdx = 0; cutsetStateIdx < cutsetSize; cutsetStateIdx += 1)
            {
                BB::FullState<DP::TSPState>& stateToDequeue = cutsets->at((cutsetMaxSize * i) + cutsetStateIdx);

                while(queue->at(activePrefixSize).active)
                {
                    activePrefixSize += 1;
                }

                if(activePrefixSize < queue->validPrefixSize)
                {
                    queue->at(activePrefixSize) = stateToDequeue;
                    activePrefixSize += 1;
                }
                else
                {
                    queue->pushBackValid(stateToDequeue);
                }

            }

            dequeuedStatesCount += cutsetSize;
        }

        checkedStatesCount += cutsetSize;
    }

    //printf("[INFO] Dequeued %u of %u states\n", dequeuedStatesCount, checkedStatesCount);

    toOffload->clearValid();

    return dequeuedStatesCount;
}
