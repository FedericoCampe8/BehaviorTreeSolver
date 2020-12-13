#include <cstdio>
#include <cinttypes>
#include <cstddef>
#include <new>
#include <utility>
#include <algorithm>
#include <fstream>

#include <Containers/StaticSet.cuh>
#include <Utils/Memory.cuh>
#include <Utils/Chrono.cuh>
#include <Utils/CUDA.cuh>
#include <External/Json.hpp>

#include "OP/TSPProblem.cuh"
#include "MDD/MDD.cuh"
#include "DP/TSPModel.cuh"
#include "BB/AugmentedState.cuh"

#define QUEUE_MAX_SIZE 1000000

using namespace std;
using json = nlohmann::json;

using AugmentedStateType = BB::AugmentedState<DP::TSPState>;
using StateType = DP::TSPState;

//Auxiliary functions
void setupGPU();
OP::TSPProblem* parseGrubHubInstance(char const * problemFileName);

//Allocation and initialization
RuntimeArray<StateType>* getArrayOfStates(OP::TSPProblem const * problem, unsigned int capacity, Memory::MallocType mallocType);
StaticVector<unsigned int>* getVectorOfUnsignedInts(unsigned int capacity, Memory::MallocType mallocType);
StaticSet<StateType>* getStaticSetOfStates(RuntimeArray<StateType>* states, StaticVector<unsigned int>* invalidStates, Memory::MallocType mallocType);
StaticVector<AugmentedStateType>* getVectorOfAugmentedStates(unsigned int capacity, Memory::MallocType mallocType);
RuntimeArray<AugmentedStateType>* getArrayOfAugmentedState(RuntimeArray<StateType>* states, Memory::MallocType mallocType);
StateType* getState(OP::TSPProblem const * problem, Memory::MallocType mallocType);


//Queue management
bool queueCmp(AugmentedStateType const& aState1, AugmentedStateType const & aState2);
bool queueChk(StateType const * bestSolution, AugmentedStateType const * aState);

//Offload management
void prepareOffloadQueue(StateType const * bestSolution, StaticSet<StateType>* mainQueueBuffer, StaticVector<AugmentedStateType>* mainQueue, RuntimeArray<StateType>* offloadQueueBuffer, StaticVector<AugmentedStateType>* offloadQueue);

void doOffload(OP::TSPProblem const * problem, unsigned int width, ManualVector<unsigned int>* toOffload, ManualVector<AugmentedStateType>* queue, RuntimeArray<StateType>* solutions, RuntimeArray<unsigned int>* cutsetsSizes, ManualVector<AugmentedStateType>* cutsets);
__global__ void offload(OP::TSPProblem const * problem, unsigned int width, ManualVector<unsigned int>* toOffload, ManualVector<AugmentedStateType>* queue, RuntimeArray<StateType>* solutions, RuntimeArray<unsigned int>* cutsetsSizes, ManualVector<AugmentedStateType>* cutsets);
void checkForBestSolutions(uint64_t startTime, uint64_t& bestSolutionTime, StateType& bestSolution, ManualVector<unsigned int>* toOffload, ManualVector<AugmentedStateType>* queue, RuntimeArray<StateType>* solutions);
unsigned int dequeueStatesFromGPU(unsigned int globalUpperBound, ManualVector<unsigned int>* toOffload, unsigned int cutsetMaxSize, RuntimeArray<unsigned int>* cutsetsSizes, ManualVector<AugmentedStateType>* cutsets, ManualVector<AugmentedStateType>* queue);

int main(int argc, char ** argv)
{
    char const * problemFileName = argv[1];
    unsigned int mddMaxWidth = std::atoi(argv[2]);
    unsigned int gpuMaxParallelism = std::atoi(argv[3]);

    setupGPU();

    auto* const problem = parseGrubHubInstance(problemFileName);

    //Main queue
    RuntimeArray<StateType>* states = getArrayOfStates(problem, QUEUE_MAX_SIZE, Memory::MallocType::Std);
    StaticVector<unsigned int>* invalidStates = getVectorOfUnsignedInts(QUEUE_MAX_SIZE, Memory::MallocType::Std);
    StaticSet<StateType>* const mainQueueBuffer = getStaticSetOfStates(states, invalidStates, Memory::MallocType::Std);
    StaticVector<AugmentedStateType>* const mainQueue = getVectorOfAugmentedStates(QUEUE_MAX_SIZE, Memory::MallocType::Std);

    //Offload queue
    RuntimeArray<StateType>* offloadQueueBuffer = getArrayOfStates(problem, gpuMaxParallelism, Memory::MallocType::Managed);
    StaticVector<AugmentedStateType>* offloadQueue = getVectorOfAugmentedStates(gpuMaxParallelism, Memory::MallocType::Managed);

    //Relaxed MDDs cutsets
    unsigned int cutsetMaxSize = MDD::MDD::calcFanout(problem) * mddMaxWidth;
    states = getArrayOfStates(problem, cutsetMaxSize * gpuMaxParallelism, Memory::MallocType::Managed);
    RuntimeArray<AugmentedStateType>* const cutsetsBuffer = getArrayOfAugmentedState(states, Memory::MallocType::Managed);
    StaticVector<unsigned int>* cutsetSizes = getVectorOfUnsignedInts(gpuMaxParallelism, Memory::MallocType::Managed);

    //Restricted MDDs bottom states
    RuntimeArray<StateType>* bottomStates = getArrayOfStates(problem, gpuMaxParallelism, Memory::MallocType::Managed);

    //Best solution
    StateType* bestSolution = getState(problem, Memory::MallocType::Managed);
    uint64_t bestSolutionTime = Chrono::now();

    //Init root
    StateType* root = getState(problem, Memory::MallocType::Std);
    DP::TSPModel::makeRoot(problem, root);

    //Enqueue root
    StateType* rootOnQueue = mainQueueBuffer->add(*root);
    mainQueue->resize(1);
    new (&mainQueue->front()) AugmentedStateType(rootOnQueue);

    unsigned int visitedStatesCount = 0;
    unsigned int iterationsCount = 0;
    auto startTime = Chrono::now();
    do
    {
        prepareOffloadQueue(bestSolution, mainQueueBuffer, mainQueue, offloadQueueBuffer, offloadQueue);
        /*
        printf("[INFO] --- Iteration %d ---\n", iterationsCount);
        printf("[INFO] Best solution: ");
        bestSolution.selectedValues.print(false);
        printf(" | Value: %d | Time %ld ms\n", bestSolution.cost, bestSolutionTime > startTime ? static_cast<long>(Chrono::now() - startTime) : 0);
        printf("[INFO] Visited states: %u\n", visitedStatesCount);
        printf("[INFO] Total time: %ld ms\n", static_cast<long>(Chrono::now() - startTime));
         */



        if (enqueuedStates > 0)
        {
            visitedStatesCount += toOffload->validPrefixSize;

            doOffload(problem, mddMaxWidth, toOffload, mainQueue, solutions, cutsetsSizes, cutsets);

            checkForBestSolutions(startTime, bestSolutionTime, bestSolution, toOffload, mainQueue, solutions);

            dequeueStatesFromGPU(bestSolution.cost, toOffload, cutsetMaxSize, cutsetsSizes, cutsets, mainQueue);

            iterationsCount += 1;
        }

        /*
        if(mainQueue->validPrefixSize >= 1000000)
        {
            //printf("[INFO] Total time: %ld ms\n", static_cast<long>(Chrono::now() - startTime));

            for (unsigned int i = 0; i < mainQueue->validPrefixSize; i += 1)
            {
                AugmentedStateType& fullState = mainQueue->at(i);
                if (fullState.active)
                {
                    printf("%u,%d\n", fullState.state->selectedValues.getSize(), fullState.state->cost);
                }
            }

            return EXIT_SUCCESS;
        }*/

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

RuntimeArray<StateType>* getArrayOfStates(OP::TSPProblem const * problem, unsigned int capacity, Memory::MallocType mallocType)
{
    assert(capacity > 0);

    //Array of states
    std::byte* arrayMem = Memory::safeMalloc(sizeof(RuntimeArray<StateType>), mallocType);
    RuntimeArray<StateType>* array = reinterpret_cast<RuntimeArray<StateType>*>(arrayMem);
    new (array) RuntimeArray<StateType>(capacity, mallocType);

    //Init states
    std::size_t stateStorageSize = StateType::sizeOfStorage(problem);
    std::byte* statesStorages = Memory::safeMalloc(stateStorageSize * capacity, mallocType);
    thrust::for_each(thrust::host, array->begin(), array->end(), [=] (StateType& state)
    {
        unsigned int stateIdx = thrust::distance(array->begin(), &state);
        new (&state) StateType(problem, &statesStorages[stateStorageSize * stateIdx]);
    });

    return array;
}

StaticVector<unsigned int>* getVectorOfUnsignedInts(unsigned int capacity, Memory::MallocType mallocType)
{
    assert(capacity > 0);

    std::byte* vectorMem = Memory::safeMalloc(sizeof(StaticVector<unsigned int>), mallocType);
    StaticVector<unsigned int>* vector = reinterpret_cast<StaticVector<unsigned int>*>(vectorMem);
    new (vector) StaticVector<unsigned int>(capacity, mallocType);

    return vector;
}

StaticSet<StateType>* getStaticSetOfStates(RuntimeArray<StateType>* states, StaticVector<unsigned int>* invalidStates, Memory::MallocType mallocType)
{
    std::byte* setMem = Memory::safeMalloc(sizeof(StaticSet<StateType>), mallocType);
    StaticSet<StateType>* set = reinterpret_cast<StaticSet<StateType>*>(setMem);
    new (set) StaticSet<StateType>(states, invalidStates);

    return set;
}

StaticVector<AugmentedStateType>* getVectorOfAugmentedStates(unsigned int capacity, Memory::MallocType mallocType)
{
    assert(capacity > 0);

    std::byte* vectorMem = Memory::safeMalloc(sizeof(StaticVector<AugmentedStateType>), mallocType);
    StaticVector<AugmentedStateType>* vector = reinterpret_cast<StaticVector<AugmentedStateType>*>(vectorMem);
    new (vector) StaticVector<AugmentedStateType>(capacity, mallocType);

    return vector;
}

RuntimeArray<AugmentedStateType>* getArrayOfAugmentedState(RuntimeArray<StateType>* states, Memory::MallocType mallocType)
{
    //Array of augmented states
    std::byte* arrayMem = Memory::safeMalloc(sizeof(RuntimeArray<AugmentedStateType>), mallocType);
    RuntimeArray<AugmentedStateType>* array = reinterpret_cast<RuntimeArray<AugmentedStateType>*>(arrayMem);
    new (array) StaticVector<AugmentedStateType>(states->getCapacity(), mallocType);

    //Init augmented states
    thrust::for_each(thrust::host, array->begin(), array->end(), [=] (AugmentedStateType& aState)
    {
        unsigned int aStateIdx = thrust::distance(array->begin(), &aState);
        new (&aState) AugmentedStateType(&states->at(aStateIdx));
    });

    return array;
}

StateType* getState(OP::TSPProblem const * problem, Memory::MallocType mallocType)
{
    //State
    std::byte* stateMem = Memory::safeMalloc(sizeof(StateType), mallocType);
    StateType* state = reinterpret_cast<StateType*>(stateMem);

    //Init
    std::size_t storageSize = StateType::sizeOfStorage(problem);
    std::byte* storage = Memory::safeMalloc(storageSize, mallocType);
    new (&state) StateType(problem, storage);

    return state;
}

StaticVector<AugmentedStateType>* getArrayOfVectorOfAugmentedState(unsigned int vectorsCapacity, RuntimeArray<AugmentedStateType> const * aStates)
{
    assert(aStates->size % vectorsCapacity == 0);

    unsigned int vectorsCount

    // Malloc
    std::byte* vectorsMem = Memory::safeManagedMalloc(sizeof(AugmentedStateType) * states->size);
    AugmentedStateType* aStates = reinterpret_cast<AugmentedStateType*>(vectorsMem);

    std::byte* arrayMem = Memory::safeManagedMalloc(sizeof(RuntimeArray<AugmentedStateType>));
    RuntimeArray<AugmentedStateType>* array = reinterpret_cast<RuntimeArray<AugmentedStateType>*>(arrayMem);

    // Init
    thrust::for_each(thrust::host, states, states + states->size, [=] (AugmentedStateType& aState)
    {
        unsigned int aStateIdx = thrust::distance(aStates, &aState);
        new (&aState) AugmentedStateType(&states->at(aStateIdx));
    });

    new (array) RuntimeArray<AugmentedStateType>(states->size, vectorsMem);

    return array;
}

bool queueCmp(AugmentedStateType const& aState1, AugmentedStateType const & aState2)
{
    return aState1.state->cost < aState2.state->cost;
}

void addToQueue(StaticSet<StateType>* queueBuffer, StaticVector<AugmentedStateType>* queue, StateType const * state)
{
   queueBuffer->add(*root);

    //Add state to queue
    unsigned int queueSize = queue->getSize();
    queue->resize(queueSize + 1);
    AugmentedStateType* stateInfo
    new (&queue->back()) AugmentedStateType(bufferEntry);
    std::push_heap(queue->begin(), queue->end(), queueCmp);
}

bool queueChk(StateType const * bestSolution, AugmentedStateType const * aState)
{
    return
        aState->lowerBound < aState->upperBound and
        aState->lowerBound < bestSolution->cost and
        aState->state->cost < bestSolution->cost;
}

void prepareOffloadQueue(StaticVector<unsigned int>* queueBufferFreeIndices, StaticVector<AugmentedStateType>* queue, StateType const * bestSolution, StaticVector<AugmentedStateType>* offloadQueue)
{
    offloadQueue->clear();
    while(not queue->isEmpty() and not offloadQueue->isFull())
    {
        //Get candidate from queue
        std::pop_heap(queue->begin(), queue->end(), queueCmp);
        AugmentedStateType* queueStateMetadata = &queue->back();


        if(queueChk(bestSolution, queueStateMetadata))
        {
            unsigned int offloadQueueSize = offloadQueue->getSize();
            offloadQueue->resize(offloadQueueSize + 1);
            AugmentedStateType* offloadStateMetadata = &offloadQueue->back();
            *offloadStateMetadata = *queueStateMetadata;
        }

        //Remove candidate buffer
        StateType* queueBuffer = queue->begin()->state;
        StateType* state = queueStateMetadata->state;
        unsigned int stateBufferIdx = thrust::distance(queueBuffer, state);
        queueBufferFreeIndices->pushBack(stateBufferIdx);

        //Remove candidate from queue
        queue->popBack();
    }
}

void doOffload(OP::TSPProblem const * problem, unsigned int width, ManualVector<unsigned int>* toOffload, ManualVector<AugmentedStateType>* queue, RuntimeArray<StateType>* solutions, RuntimeArray<unsigned int>* cutsetsSizes, ManualVector<AugmentedStateType>* cutsets)
{
        offload<<<toOffload->validPrefixSize, 32>>>(problem, width, toOffload, queue, solutions, cutsetsSizes, cutsets);
        cudaDeviceSynchronize();
}

__global__
void offload(OP::TSPProblem const * problem, unsigned int width, ManualVector<unsigned int>* toOffload, ManualVector<AugmentedStateType>* queue, RuntimeArray<StateType>* solutions, RuntimeArray<unsigned int>* cutsetsSizes, ManualVector<AugmentedStateType>* cutsets)
{
    __shared__ unsigned int alignedSharedMem[1000];
    std::byte* sharedMem = reinterpret_cast<std::byte*>(alignedSharedMem);

    ONE_THREAD_IN_BLOCK
    {
        //MDD
        unsigned int fullStateIdx = toOffload->at(blockIdx.x);
        AugmentedStateType& enqueuedState = queue->at(fullStateIdx);
        MDD::MDD mdd(MDD::MDD::Type::Relaxed, width, enqueuedState.state, problem);

        StateType* solution = &solutions->at(blockIdx.x);
        unsigned int cutsetMaxSize = width * MDD::MDD::calcFanout(problem);
        unsigned int& cutsetSize = cutsetsSizes->at(blockIdx.x);

        AugmentedStateType* cutset = &cutsets->at(cutsetMaxSize * blockIdx.x);
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

void checkForBestSolutions(uint64_t startTime, uint64_t& bestSolutionTime, StateType& bestSolution, ManualVector<unsigned int>* toOffload, ManualVector<AugmentedStateType>* queue, RuntimeArray<StateType>* solutions)
{
    for(unsigned int i = 0; i < toOffload->validPrefixSize; i += 1)
    {
        unsigned int fullStateIdx = toOffload->at(i);
        AugmentedStateType& fullState = queue->at(fullStateIdx);
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

unsigned int dequeueStatesFromGPU(unsigned int globalUpperBound, ManualVector<unsigned int>* toOffload, unsigned int cutsetMaxSize, RuntimeArray<unsigned int>* cutsetsSizes, ManualVector<AugmentedStateType>* cutsets, ManualVector<AugmentedStateType>* queue)
{
    unsigned int checkedStatesCount = 0;
    unsigned int dequeuedStatesCount = 0;
    unsigned int activePrefixSize = 0;

    for (unsigned int i = 0; i < toOffload->validPrefixSize; i += 1)
    {
        unsigned int enqueuedStateIdx = toOffload->at(i);
        AugmentedStateType& enqueuedState = queue->at(enqueuedStateIdx);
        unsigned int cutsetSize = cutsetsSizes->at(i);

        if(not enqueuedState.state->selectedValues.isFull())
        {
            for (unsigned int cutsetStateIdx = 0; cutsetStateIdx < cutsetSize; cutsetStateIdx += 1)
            {
                AugmentedStateType& stateToDequeue = cutsets->at((cutsetMaxSize * i) + cutsetStateIdx);

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
