#include <cstdio>
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

#define CPU_QUEUE_SIZE 150000

using namespace std;
using json = nlohmann::json;

//Auxiliary functions
void setupGPU();

OP::TSPProblem* parseGrubHubInstance(char const * problemFileName);

ManualVector<BB::FullState<DP::TSPState>>* fullStateBuffer(OP::TSPProblem const * problem, unsigned int size);
RuntimeArray<DP::TSPState>* stateArray(OP::TSPProblem const * problem, unsigned int size);
StaticVector<unsigned int>* uintVector(unsigned int size);
RuntimeArray<unsigned int>* uintArray(unsigned int size);

void prepareOffload(int& gUpperBound, ManualVector<BB::FullState<DP::TSPState>>* queue, StaticVector<unsigned int>* toOffload);
__global__ void doOffload(OP::TSPProblem const * problem, unsigned int width, StaticVector<unsigned int>* toOffload, ManualVector<BB::FullState<DP::TSPState>>* queue, RuntimeArray<DP::TSPState>* solutions, RuntimeArray<unsigned int>* cutsetsSizes, ManualVector<BB::FullState<DP::TSPState>>* cutsets);

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
    StaticVector<unsigned int>* const toOffload = uintVector(maxParallelism);

    // Root
    DP::TSPModel::makeRoot(problem, queue->at(0).state);
    queue->at(0).upperBound = INT32_MAX;
    queue->at(0).lowerBound = INT32_MIN;
    queue->validPrefixSize = 1;

    //Solutions
    RuntimeArray<DP::TSPState>* solutions = stateArray(problem, maxParallelism + 1);
    DP::TSPState& bestSolution = solutions->at(solutions->size - 1);

    //Bound
    int32_t gUpperBound = INT32_MAX;

    auto start = Chrono::now();
    do
    {
        printf("[INFO] Queue size: %d\r", queue->validPrefixSize );

        // Offload
        prepareOffload(gUpperBound, queue, toOffload);

        if (not toOffload->isEmpty())
        {
            {
                auto start = Chrono::now();
                doOffload<<<toOffload->getSize(), 32>>>(problem, width, toOffload, queue, solutions, cutsetsSizes, cutsets);
                cudaDeviceSynchronize();
                auto end = Chrono::now();
                //printf("[INFO] Explored %d states on GPU (%d ms)\n", toOffload->getSize(), static_cast<unsigned int>(end - start));
            }

            // Update global upper bound
            for (unsigned int i = 0; i < toOffload->getSize(); i += 1)
            {
                unsigned int fullStateIdx = toOffload->at(i);
                BB::FullState<DP::TSPState>& fullState = queue->at(fullStateIdx);
                if (fullState.upperBound < gUpperBound)
                {
                    gUpperBound = fullState.upperBound;
                    bestSolution = solutions->at(i);

                    auto end = Chrono::now();
                    printf("[INFO] Found new solution ");
                    bestSolution.selectedValues.print(false);
                    printf(" of value %d in %d ms\n", gUpperBound, static_cast<unsigned int>(end - start));
                }
            }

            // Filter useless state
            unsigned int filteredStates = 0;
            unsigned int enquequedStates = 0;
            unsigned int totalStates = 0;
            for (unsigned int i = 0; i < toOffload->getSize(); i += 1)
            {
                unsigned int& cutsetSize = cutsetsSizes->at(i);
                totalStates += cutsetSize;

                unsigned int fullStateIdx = toOffload->at(i);
                BB::FullState<DP::TSPState>& fullState = queue->at(fullStateIdx);
                if (fullState.lowerBound > gUpperBound or fullState.state->selectedValues.getSize() == problem->vars.size)
                {
                    //printf("[INFO] Filtering cutset %d of size %d\n", i, cutsetSize);
                    filteredStates += cutsetSize;
                    cutsetSize = 0;
                }
            }
            //printf("[INFO] Dequeue filtered %d/%d states\n", filteredStates, totalStates);


            // Push cutsets
            for (unsigned int i = 0; i < toOffload->getSize(); i += 1)
            {
                unsigned int& cutsetSize = cutsetsSizes->at(i);
                if (cutsetSize > 0)
                {
                    unsigned int fullStateIdx = toOffload->at(i);
                    BB::FullState<DP::TSPState>& fullState = queue->at(fullStateIdx);
                    //printf("[INFO] Pushing cutset %d of size %d\n", i, cutsetSize);
                    for (unsigned int j = 0; j < cutsetSize; j += 1)
                    {
                        queue->at(queue->validPrefixSize) = cutsets->at((cutsetMaxSize * i) + j);
                        queue->validPrefixSize += 1;

                        enquequedStates += 1;

                        assert(queue->validPrefixSize <= CPU_QUEUE_SIZE);
                    }
                }
            }
            //printf("[INFO] Dequeue pushed %d/%d states\n", enquequedStates, totalStates);

            toOffload->clear();
        }
    }
    while(queue->validPrefixSize > 0);

    auto end = Chrono::now();

    printf("[INFO] Found optimal solution ");
    bestSolution.selectedValues.print(false);
    printf(" of value %d in %d ms\n", gUpperBound, static_cast<unsigned int>(end - start));

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
        fullStates[fullStateIdx].state = &states->at(fullStateIdx);
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

StaticVector<unsigned int>* uintVector(unsigned int size)
{
    // Malloc
    std::byte* storage = Memory::safeManagedMalloc(StaticVector<unsigned int>::sizeofStorage(size));

    std::byte* vectorMem = Memory::safeManagedMalloc(sizeof(StaticVector<unsigned int>));
    StaticVector<unsigned int>* vector = reinterpret_cast<StaticVector<unsigned int>*>(vectorMem);

    // Init
    new (vector) StaticVector<unsigned int>(size, storage);

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


void prepareOffload(int& gUpperBound, ManualVector<BB::FullState<DP::TSPState>>* queue, StaticVector<unsigned int>* toOffload)
{
    unsigned int filteredStates = 0;
    while(queue->validPrefixSize > 0 and not toOffload->isFull())
    {
        BB::FullState<DP::TSPState>& fullState = queue->lastValid();

        if(fullState.lowerBound < gUpperBound)
        {
            toOffload->pushBack(queue->validPrefixSize - 1);
        }
        else
        {
            filteredStates +=1;
        }

        queue->validPrefixSize -= 1;
    }
    //printf("[INFO] Enqueue filtered %d states\n", filteredStates);
}

__global__
void doOffload(OP::TSPProblem const * problem, unsigned int width, StaticVector<unsigned int>* toOffload, ManualVector<BB::FullState<DP::TSPState>>* queue, RuntimeArray<DP::TSPState>* solutions, RuntimeArray<unsigned int>* cutsetsSizes, ManualVector<BB::FullState<DP::TSPState>>* cutsets)
{
    __shared__ unsigned int alignedSharedMem[1000];
    std::byte* sharedMem = reinterpret_cast<std::byte*>(alignedSharedMem);

    ONE_THREAD_IN_BLOCK
    {

        //MDD
        auto start = Chrono::now();

        BB::FullState<DP::TSPState>& queueFullState = queue->at(toOffload->at(blockIdx.x));
        MDD::MDD mdd(MDD::MDD::Type::Relaxed, width, queueFullState.state, problem);

        DP::TSPState* solution = &solutions->at(blockIdx.x);
        unsigned int cutsetMaxSize = width * MDD::MDD::calcFanout(problem);
        unsigned int& cutsetSize = cutsetsSizes->at(blockIdx.x);

        BB::FullState<DP::TSPState>* cutset = &cutsets->at(cutsetMaxSize * blockIdx.x);
        mdd.buildTopDown(solution, cutsetSize, cutset->state, sharedMem);
        queueFullState.lowerBound = solution->cost;

        mdd.type = MDD::MDD::Type::Restricted;
        mdd.buildTopDown(solution, cutsetSize, cutset->state, sharedMem);
        queueFullState.upperBound = solution->cost;

        for(unsigned int cutsetStateIdx = 0; cutsetStateIdx < cutsetSize; cutsetStateIdx +=1)
        {
            cutset[cutsetStateIdx].lowerBound = queueFullState.lowerBound;
            cutset[cutsetStateIdx].upperBound = queueFullState.upperBound;
        }

        auto end = Chrono::now();

        /*
        printf("[INFO] == State\n");
        if (queueFullState.upperBound < 2802)
        {
            mdd.print(queueFullState.level);
        }
        printf("[INFO] lowerbound = %d\n", queueFullState.lowerBound);
        printf("[INFO] upperbound = %d\n", queueFullState.upperBound);
        printf("[INFO] level = %d\n", queueFullState.level);
        printf("[INFO] ");
        queueFullState.state->selectedValues.print(false);
        printf(" - %d - ", queueFullState.level);
        queueFullState.state->admissibleValues.print();

        printf("[INFO] == Cuteset\n");
        for(int i = 0; i < queueFullState.cutsetSize; i +=1)
        {
            if (not cutset[i].state->active)
            {
                break;
            }
            printf("[INFO] ");
            cutset[i].state->selectedValues.print(false);
            printf(" - %d - ", cutset[i].level);
            cutset[i].state->admissibleValues.print();
        }
         */

        //printf("[INFO] State %d with bounds (%d,%d) and cutset %d explored in %d ms\n", blockIdx.x,  queueFullState.lowerBound, queueFullState.upperBound, cutsetSize, static_cast<unsigned int>(end - start));


    }
}