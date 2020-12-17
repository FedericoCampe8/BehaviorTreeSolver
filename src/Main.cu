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
#include <External/Json.hpp>

#include "OP/VRProblem.cuh"
#include "DP/VRPModel.cuh"
#include "BB/AugmentedState.cuh"
#include "DP/VRPState.cuh"

using namespace std;
using json = nlohmann::json;

using AugmentedStateType = BB::AugmentedState<DP::VRPState>;
using StateType = DP::VRPState;

//Auxiliary functions
void setupGPU();
OP::VRProblem* parseGrubHubInstance(char const * problemFileName, Memory::MallocType mallocType);

//Allocation and initialization
RuntimeArray<StateType>* getArrayOfStates(OP::VRProblem const * problem, unsigned int capacity, Memory::MallocType mallocType);
StaticVector<unsigned int>* getVectorOfUnsignedInts(unsigned int capacity, Memory::MallocType mallocType);
StaticSet<StateType>* getStaticSetOfStates(RuntimeArray<StateType>* states, StaticVector<unsigned int>* invalidStates, Memory::MallocType mallocType);
StaticVector<AugmentedStateType>* getVectorOfAugmentedStates(unsigned int capacity, Memory::MallocType mallocType);
RuntimeArray<AugmentedStateType>* getArrayOfAugmentedState(RuntimeArray<StateType>* states, Memory::MallocType mallocType);
RuntimeArray<unsigned int>* getArrayOfUnsignedInts(unsigned int capacity, Memory::MallocType mallocType);
StateType* getState(OP::VRProblem const * problem, Memory::MallocType mallocType);

//Main queue
bool queueCmp(AugmentedStateType const& aState1, AugmentedStateType const & aState2);
bool queueChk(StateType const * bestSolution, AugmentedStateType const & aState);
void updateMainQueue(StateType const * bestSolution, StaticSet<StateType>* mainQueueBuffer, StaticVector<AugmentedStateType>* mainQueue, StaticVector<AugmentedStateType>* offloadQueue, unsigned int cutsetMaxSize, RuntimeArray<unsigned int>* cutsetsSizes, RuntimeArray<AugmentedStateType>* cutsetsBuffer);
void reduceMainQueue(OP::VRProblem const * problem, unsigned int mainQueueReducedSize, StaticSet<StateType>* mainQueueBuffer, StaticVector<AugmentedStateType>* mainQueue, StaticSet<StateType>* shadowQueueBuffer, StaticVector<AugmentedStateType>* shadowQueue);

//Offload
void prepareOffloadQueue(StateType const * bestSolution, StaticSet<StateType>* mainQueueBuffer, StaticVector<AugmentedStateType>* mainQueue, RuntimeArray<StateType>* offloadQueueBuffer, StaticVector<AugmentedStateType>* offloadQueue);
__global__ void offloadGPU(OP::VRProblem const * problem, unsigned int mddMaxWidth, RuntimeArray<StateType>* offloadQueueBuffer, StaticVector<AugmentedStateType>* offloadQueue, unsigned int cutsetMaxSize, RuntimeArray<unsigned int>* cutsetsSizes, RuntimeArray<AugmentedStateType>* cutsetsBuffer, RuntimeArray<StateType>* bottomStatesBuffer);
void offloadCPU(OP::VRProblem const * problem, unsigned int mddMaxWidth, RuntimeArray<StateType>* offloadQueueBuffer, StaticVector<AugmentedStateType>* offloadQueue, unsigned int cutsetMaxSize, RuntimeArray<unsigned int>* cutsetsSizes, RuntimeArray<AugmentedStateType>* cutsetsBuffer, RuntimeArray<StateType>* bottomStatesBuffer, std::byte* scratchpadMem);
__host__ __device__ void offload(OP::VRProblem const * problem, unsigned int mddMaxWidth, RuntimeArray<StateType>* offloadQueueBuffer, StaticVector<AugmentedStateType>* offloadQueue, unsigned int cutsetMaxSize, RuntimeArray<unsigned int>* cutsetsSizes, RuntimeArray<AugmentedStateType>* cutsetsBuffer, RuntimeArray<StateType>* bottomStatesBuffer, std::byte* scratchpadMem, unsigned int idx);

//Search
bool checkForBetterSolutions(StateType* bestSolution, StaticVector<AugmentedStateType>* offloadQueue, RuntimeArray<StateType>* bottomStatesBuffer);

//Debug
void printElapsedTime(uint64_t elapsedTimeMs);
void printQueue(StaticVector<AugmentedStateType>* queue);
void printCutsets(StaticVector<AugmentedStateType>* offloadQueue, unsigned int cutsetMaxSize, RuntimeArray<unsigned int>* cutsetsSizes, RuntimeArray<AugmentedStateType>* cutsetsBuffer);

int main(int argc, char ** argv)
{
    /*
    char const * problemFileName = argv[1];
    unsigned int timeoutSeconds = std::stoi(argv[2]);
    unsigned int mainQueueMaxSize = std::stoul(argv[3]);
    unsigned int mainQueueReducedSize = std::stoul(argv[4]);
    unsigned int mddMaxWidth = std::stoi(argv[5]);
    unsigned int maxParallelism = std::stoi(argv[6]);
    char propagationType = *argv[7];

    Memory::MallocType offloadInvolvedMallocType = propagationType == 'g' ? Memory::MallocType::Managed : Memory::MallocType::Std;
    std::byte* scratchpadMem = Memory::safeStdMalloc(MDD_SCRATCHPAD_SIZE * maxParallelism);
    if(propagationType == 'g')
    {
        setupGPU();
    }

    OP::VRProblem* problem = parseGrubHubInstance(problemFileName, offloadInvolvedMallocType);

    //Main queue
    RuntimeArray<StateType>* states = getArrayOfStates(problem, mainQueueMaxSize, Memory::MallocType::Std);
    StaticVector<unsigned int>* invalidStates = getVectorOfUnsignedInts(mainQueueMaxSize, Memory::MallocType::Std);
    StaticSet<StateType>* mainQueueBuffer = getStaticSetOfStates(states, invalidStates, Memory::MallocType::Std);
    StaticVector<AugmentedStateType>* mainQueue = getVectorOfAugmentedStates(mainQueueMaxSize, Memory::MallocType::Std);

    //Shadow queue
    states = getArrayOfStates(problem, mainQueueMaxSize, Memory::MallocType::Std);
    invalidStates = getVectorOfUnsignedInts(mainQueueMaxSize, Memory::MallocType::Std);
    StaticSet<StateType>* shadowQueueBuffer = getStaticSetOfStates(states, invalidStates, Memory::MallocType::Std);
    StaticVector<AugmentedStateType>* shadowQueue = getVectorOfAugmentedStates(mainQueueMaxSize, Memory::MallocType::Std);

    //Offload queue
    RuntimeArray<StateType>* offloadQueueBuffer = getArrayOfStates(problem, maxParallelism, offloadInvolvedMallocType);
    StaticVector<AugmentedStateType>* offloadQueue = getVectorOfAugmentedStates(maxParallelism, offloadInvolvedMallocType);

    //Relaxed MDDs cutsets
    unsigned int cutsetMaxSize = MDD::calcFanout(problem) * mddMaxWidth;
    states = getArrayOfStates(problem, cutsetMaxSize * maxParallelism, offloadInvolvedMallocType);
    RuntimeArray<AugmentedStateType>* const cutsetsBuffer = getArrayOfAugmentedState(states, offloadInvolvedMallocType);
    RuntimeArray<unsigned int>* cutsetsSizes = getArrayOfUnsignedInts(maxParallelism, offloadInvolvedMallocType);

    //Restricted MDDs bottom states
    RuntimeArray<StateType>* bottomStatesBuffer = getArrayOfStates(problem, maxParallelism, offloadInvolvedMallocType);

    //Solution
    StateType* bestSolution = getState(problem, offloadInvolvedMallocType);
    bestSolution->cost = UINT32_MAX;

    //Init root
    StateType* root = getState(problem, offloadInvolvedMallocType);
    DP::VRPModel::makeRoot(problem, root);

    //Enqueue root
    StateType* rootOnQueue = mainQueueBuffer->insert(*root);
    mainQueue->resize(1);
    new (&mainQueue->back()) AugmentedStateType(rootOnQueue);
    mainQueue->back().upperBound = UINT32_MAX;

    //Search
    assert(mainQueueReducedSize + (cutsetMaxSize * maxParallelism) <= mainQueueMaxSize);
    unsigned int visitedStatesCount = 0;
    unsigned int iterationsCount = 0;
    uint64_t startTime = Chrono::now();
    do
    {
        if (mainQueue->getSize() + (cutsetMaxSize * maxParallelism) > mainQueueMaxSize)
        {
            //printf("[DEBUG] Queque: %u -> ", mainQueue->getSize());
            reduceMainQueue(problem, mainQueueReducedSize, mainQueueBuffer, mainQueue, shadowQueueBuffer, shadowQueue);
            //printf("%u                                                                                               \n", mainQueue->getSize());
        }

        prepareOffloadQueue(bestSolution, mainQueueBuffer, mainQueue, offloadQueueBuffer, offloadQueue);

        printf("[DEBUG] Offload queue: ");
        printQueue(offloadQueue);

        if (not offloadQueue->isEmpty())
        {

            uint64_t offloadStartTime = Chrono::now();
            if(propagationType == 'g')
            {
                offloadGPU<<<offloadQueue->getSize(),1>>>(problem, mddMaxWidth, offloadQueueBuffer, offloadQueue, cutsetMaxSize, cutsetsSizes, cutsetsBuffer, bottomStatesBuffer);
                cudaDeviceSynchronize();
            }
            else
            {
                offloadCPU(problem, mddMaxWidth, offloadQueueBuffer, offloadQueue, cutsetMaxSize, cutsetsSizes, cutsetsBuffer, bottomStatesBuffer, scratchpadMem);
            }

            printf("[DEBUG] Cutsets: ");
            printCutsets(offloadQueue, cutsetMaxSize, cutsetsSizes, cutsetsBuffer);

            visitedStatesCount += offloadQueue->getSize();

            if(checkForBetterSolutions(bestSolution, offloadQueue, bottomStatesBuffer))
            {
                printf("[INFO] Better solution found: ");
                bestSolution->selectedValues.print(false);
                printf(" | Value: %u", bestSolution->cost);
                printf(" | Time: ");
                printElapsedTime(Chrono::now() - startTime);
                printf(" | Iterations: %u", iterationsCount);
                printf(" | Visited states: %u\n", visitedStatesCount);
            }
            else
            {
                printf("[INFO] Speed: %5lu states/s", static_cast<uint64_t>(offloadQueue->getSize()) * 1000 / (Chrono::now() - offloadStartTime));
                printf(" | Time: ");
                printElapsedTime(Chrono::now() - startTime);
                printf(" | Iterations: %u", iterationsCount);
                printf(" | State to visit: %u", mainQueue->getSize());
                printf(" | Visited states: %u\r", visitedStatesCount);
            }

            updateMainQueue(bestSolution, mainQueueBuffer, mainQueue, offloadQueue, cutsetMaxSize, cutsetsSizes, cutsetsBuffer);

            printf("[DEBUG] Main queue: ");
            printQueue(mainQueue);

            iterationsCount += 1;
        }
    }
    while((not offloadQueue->isEmpty()) and ((Chrono::now() - startTime) < timeoutSeconds * 1000));

    printf("[RESULT] Solution: ");
    bestSolution->selectedValues.print(false);
    printf(" | Value: %u", bestSolution->cost);
    printf(" | Time: ");
    printElapsedTime(Chrono::now() - startTime);
    printf(" | Iterations: %u", iterationsCount);
    printf(" | Visited states: %u\n", visitedStatesCount);
    */

    return EXIT_SUCCESS;
}

void setupGPU()
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
    unsigned int const variablesCount = problemJson["nodes"].size();
    unsigned int const problemSize = sizeof(sizeof(OP::VRProblem));
    unsigned int const problemStorageSize = OP::VRProblem::sizeOfStorage(variablesCount);
    std::byte* const memory = Memory::safeMalloc(problemSize + problemStorageSize, mallocType);

    // Init problem
    OP::VRProblem* const problem = reinterpret_cast<OP::VRProblem*>(memory);
    std::byte* const problemStorage = &memory[problemSize];
    new (problem) OP::VRProblem(variablesCount, problemStorage);

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

bool queueCmp(AugmentedStateType const& aState1, AugmentedStateType const & aState2)
{
    unsigned int w1 =  aState1.state->cost;
    unsigned int w2 =  aState2.state->cost;

    return w1 > w2;
 }

bool queueChk(StateType const * bestSolution, AugmentedStateType const & aState)
{
    return
        aState.lowerBound < aState.upperBound and
        aState.lowerBound < bestSolution->cost and
        aState.state->cost < bestSolution->cost;
}

void prepareOffloadQueue(StateType const * bestSolution, StaticSet<StateType>* mainQueueBuffer, StaticVector<AugmentedStateType>* mainQueue, RuntimeArray<StateType>* offloadQueueBuffer, StaticVector<AugmentedStateType>* offloadQueue)
{
    offloadQueue->clear();
    while(not (mainQueue->isEmpty() or offloadQueue->isFull()))
    {
        //Get state from main queue
        std::pop_heap(mainQueue->begin(), mainQueue->end(), queueCmp);
        AugmentedStateType& mainQueueAugmentedState = mainQueue->back();
        if(queueChk(bestSolution, mainQueueAugmentedState))
        {
            //Copy state to offload queue
            offloadQueue->resize(offloadQueue->getSize() + 1);
            unsigned int offloadQueueIdx = offloadQueue->getSize() - 1;
            StateType* offloadQueueState = &offloadQueueBuffer->at(offloadQueueIdx);
            *offloadQueueState = *mainQueueAugmentedState.state;
            AugmentedStateType* offloadQueueAugmentedState = &offloadQueue->at(offloadQueueIdx);
            new (offloadQueueAugmentedState) AugmentedStateType(mainQueueAugmentedState.lowerBound, mainQueueAugmentedState.upperBound, offloadQueueState);
        }

        //Remove state from main queue
        mainQueueBuffer->erase(mainQueueAugmentedState.state);
        mainQueue->popBack();
    }
}

void reduceMainQueue(OP::VRProblem const * problem, unsigned int mainQueueReducedSize, StaticSet<StateType>* mainQueueBuffer, StaticVector<AugmentedStateType>* mainQueue, StaticSet<StateType>* shadowQueueBuffer, StaticVector<AugmentedStateType>* shadowQueue)
{
    auto reduceCmp = [=](AugmentedStateType const& aState1, AugmentedStateType const& aState2) -> bool
    {
        unsigned int level1 = aState1.state->selectedValues.getSize();
        unsigned int level2 = aState2.state->selectedValues.getSize();

        if (level1 < level2)
        {
            return true;
        }
        else if (level1 == level2)
        {
            unsigned int cost1 = aState1.state->cost;
            unsigned int cost2 = aState2.state->cost;

            return cost1 < cost2;
        }
        else
        {
            return false;
        }
    };

    std::sort(mainQueue->begin(), mainQueue->end(), reduceCmp);

    RuntimeArray<unsigned int> levelsSizes(problem->variables.getCapacity(), Memory::MallocType::Std);
    thrust::fill(levelsSizes.begin(), levelsSizes.end(), 0);

    for (unsigned int i = 0; i < mainQueue->getSize(); i += 1)
    {
        unsigned int stateLevel = mainQueue->at(i).state->selectedValues.getSize();
        levelsSizes.at(stateLevel) += 1;
    }

    RuntimeArray<unsigned int> levelsBegins(problem->variables.getCapacity(), Memory::MallocType::Std);
    RuntimeArray<unsigned int> levelsEnds(problem->variables.getCapacity(), Memory::MallocType::Std);

    levelsBegins.at(0) = 0;
    levelsEnds.at(0) = levelsSizes.at(0);
    for (unsigned int i = 1; i < levelsSizes.getCapacity(); i += 1)
    {
        levelsBegins.at(i) = levelsEnds.at(i - 1);
        levelsEnds.at(i) = levelsBegins.at(i) + levelsSizes.at(i);
    }

    shadowQueueBuffer->clear();
    shadowQueue->clear();

    unsigned int notEmptyLevels = 0;
    for (unsigned int i = 1; i < levelsSizes.getCapacity(); i += 1)
    {
        if (levelsSizes.at(i) > 0)
        {
            notEmptyLevels += 1;
        }
    }

    unsigned int elementsPerLevel = mainQueueReducedSize / notEmptyLevels;
    for (unsigned int i = 1; i < levelsSizes.getCapacity(); i += 1)
    {
        unsigned int levelSize = levelsSizes.at(i);
        if (levelSize > 0)
        {
            unsigned int step = max(1, levelSize / elementsPerLevel);
            for (unsigned int stateIdx = levelsBegins.at(i); stateIdx < levelsEnds.at(i); stateIdx += step)
            {
                AugmentedStateType& stateOnMainQueue = mainQueue->at(stateIdx);
                StateType* stateOnShadowQueueBuffer = shadowQueueBuffer->insert(*stateOnMainQueue.state);
                shadowQueue->resize(shadowQueue->getSize() + 1);
                new(&shadowQueue->back()) AugmentedStateType(stateOnMainQueue.lowerBound,stateOnMainQueue.upperBound, stateOnShadowQueueBuffer);
                std::push_heap(shadowQueue->begin(), shadowQueue->end(), queueCmp);
            }
        }
    }

    thrust::swap(mainQueueBuffer,*shadowQueueBuffer);
    thrust::swap(mainQueue,shadowQueue);
}

__global__
void offloadGPU(OP::VRProblem const * problem, unsigned int mddMaxWidth, RuntimeArray<StateType>* offloadQueueBuffer, StaticVector<AugmentedStateType>* offloadQueue, unsigned int cutsetMaxSize, RuntimeArray<unsigned int>* cutsetsSizes, RuntimeArray<AugmentedStateType>* cutsetsBuffer, RuntimeArray<StateType>* bottomStatesBuffer)
{
    __shared__ unsigned int alignedSharedMem[500];
    std::byte* scratchpadMem = reinterpret_cast<std::byte*>(alignedSharedMem);

    if(blockIdx.x * blockDim.x + threadIdx.x == 0)
    {
        offload(problem, mddMaxWidth, offloadQueueBuffer, offloadQueue, cutsetMaxSize, cutsetsSizes, cutsetsBuffer, bottomStatesBuffer, scratchpadMem,  blockIdx.x);
    };
}

void offloadCPU(OP::VRProblem const * problem, unsigned int mddMaxWidth, RuntimeArray<StateType>* offloadQueueBuffer, StaticVector<AugmentedStateType>* offloadQueue, unsigned int cutsetMaxSize, RuntimeArray<unsigned int>* cutsetsSizes, RuntimeArray<AugmentedStateType>* cutsetsBuffer, RuntimeArray<StateType>* bottomStatesBuffer, std::byte* scratchpadMem)
{
    thrust::for_each(thrust::host, offloadQueue->begin(), offloadQueue->end(), [&](AugmentedStateType& offloadedState)
    {
        unsigned int idx = thrust::distance(offloadQueue->begin(), &offloadedState);
        offload(problem, mddMaxWidth, offloadQueueBuffer, offloadQueue, cutsetMaxSize, cutsetsSizes, cutsetsBuffer, bottomStatesBuffer, &scratchpadMem[100 * idx], idx);
    });
}

__host__ __device__
void offload(OP::VRProblem const * problem, unsigned int mddMaxWidth, RuntimeArray<StateType>* offloadQueueBuffer, StaticVector<AugmentedStateType>* offloadQueue, unsigned int cutsetMaxSize, RuntimeArray<unsigned int>* cutsetsSizes, RuntimeArray<AugmentedStateType>* cutsetsBuffer, RuntimeArray<StateType>* bottomStatesBuffer, std::byte* scratchpadMem, unsigned int idx)
{
    //Preparation to build MDDs
    StateType& top = offloadQueueBuffer->at(idx);
    unsigned int& cutsetSize = cutsetsSizes->at(idx);
    StateType* cutset = cutsetsBuffer->at(cutsetMaxSize * idx).state;
    StateType& bottom = bottomStatesBuffer->at(idx);
    uint32_t& lowerBound = offloadQueue->at(idx).lowerBound;
    uint32_t& upperBound = offloadQueue->at(idx).upperBound;

    //Build MDDs
q    MDD::buildMddTopDown(problem, mddMaxWidth, MDD::MDDType::Relaxed, top, cutsetMaxSize, cutsetSize, cutset, bottom, scratchpadMem);
    lowerBound = bottom.cost;
    MDD::buildMddTopDown(problem, mddMaxWidth, MDD::MDDType::Restricted, top, cutsetMaxSize, cutsetSize, cutset, bottom, scratchpadMem);
    upperBound = bottom.cost;

    //Adjust cutsets
    for(unsigned int cutsetStateIdx = 0; cutsetStateIdx < cutsetSize; cutsetStateIdx += 1)
    {
        AugmentedStateType& aState = cutsetsBuffer->at(cutsetStateIdx);
        aState.lowerBound = lowerBound;
        aState.upperBound = upperBound;
    }
}

bool checkForBetterSolutions(StateType* bestSolution, StaticVector<AugmentedStateType>* offloadQueue, RuntimeArray<StateType>* bottomStatesBuffer)
{
    bool foundBetterSolution = false;

    for(unsigned int aStateIdx = 0; aStateIdx < offloadQueue->getSize(); aStateIdx += 1)
    {
        if (offloadQueue->at(aStateIdx).upperBound < bestSolution->cost)
        {
            *bestSolution = bottomStatesBuffer->at(aStateIdx);
            foundBetterSolution = true;
        }
    }

    return foundBetterSolution;
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


void printQueue(StaticVector<AugmentedStateType>* queue)
{
    if(not queue->isEmpty())
    {
        queue->at(0).state->selectedValues.print(false);
        for(unsigned int i = 1; i < queue->getSize(); i += 1)
        {
            printf(",");
            queue->at(i).state->selectedValues.print(false);
        }
    }
    printf("\n");
}

void printCutsets(StaticVector<AugmentedStateType>* offloadQueue, unsigned int cutsetMaxSize, RuntimeArray<unsigned int>* cutsetsSizes, RuntimeArray<AugmentedStateType>* cutsetsBuffer)
{
    if(not offloadQueue->isEmpty())
    {
        for(unsigned int idx = 0; idx < offloadQueue->getSize(); idx += 1)
        {
            unsigned int cutsetSize = cutsetsSizes->at(idx);
            if(cutsetSize > 0)
            {
                AugmentedStateType* cutset = &cutsetsBuffer->at(cutsetMaxSize * idx);
                cutset[0].state->selectedValues.print(false);
                for (unsigned int cutsetStateIdx = 1; cutsetStateIdx < cutsetSize; cutsetStateIdx += 1)
                {
                    printf(",");
                    cutset[cutsetStateIdx].state->selectedValues.print(false);
                }
            }
        }
    }
    printf("\n");
}

void updateMainQueue(StateType const * bestSolution, StaticSet<StateType>* mainQueueBuffer, StaticVector<AugmentedStateType>* mainQueue, StaticVector<AugmentedStateType>* offloadQueue, unsigned int cutsetMaxSize, RuntimeArray<unsigned int>* cutsetsSizes, RuntimeArray<AugmentedStateType>* cutsetsBuffer)
{

    assert(mainQueue->getSize() + (cutsetMaxSize * offloadQueue->getCapacity()) < mainQueue->getCapacity());

    for(unsigned int idx = 0; idx < offloadQueue->getSize(); idx += 1)
    {
        AugmentedStateType* cutsetBegin = &cutsetsBuffer->at(cutsetMaxSize * idx);
        AugmentedStateType* cutsetEnd = cutsetBegin + cutsetsSizes->at(idx);
        for(AugmentedStateType* cutsetState = cutsetBegin; cutsetState != cutsetEnd; cutsetState +=1)
        {
            if(queueChk(bestSolution, *cutsetState))
            {
                StateType* cutsetStateOnQueue = mainQueueBuffer->insert(*cutsetState->state);
                mainQueue->resize(mainQueue->getSize() + 1);
                new (&mainQueue->back()) AugmentedStateType(cutsetState->lowerBound, cutsetState->upperBound, cutsetStateOnQueue);
                std::push_heap(mainQueue->begin(), mainQueue->end(), queueCmp);
            }
        }
    }
}