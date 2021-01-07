#include <thrust/binary_search.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <Utils/Chrono.cuh>
#include "DP/VRPModel.cuh"

using namespace std;
using namespace Memory;
using namespace Chrono;
using namespace DP;
using namespace OP;
using ProblemType = VRProblem;
using StateType = VRPState;

// Auxiliary functions
void configGPU();

// Debug
void printElapsedTime(uint64_t elapsedTimeMs);

void clearLine();

int main(int argc, char ** argv)
{
    // Input parsing
    char const * problemFileName = argv[1];
    unsigned int const maxWidth = std::stoi(argv[2]);

    // *******************
    // Data initialization
    // *******************

    // Context initialization
    MallocType gpuMallocType = MallocType::Managed;
    configGPU();

    // Problem
    ProblemType* const problem = VRProblem::parseGrubHubInstance(problemFileName, gpuMallocType);

    // Current states
    unsigned int memorySize = sizeof(Vector<StateType>);
    byte* memory = safeMalloc(memorySize, gpuMallocType);
    Vector<StateType> * const currentStates = new (memory) Vector<StateType>(maxWidth, gpuMallocType);
    memory = StateType::mallocStorages(problem, maxWidth, gpuMallocType);
    memorySize = StateType::sizeOfStorage(problem);
    for(unsigned int stateIdx = 0; stateIdx < currentStates->getCapacity(); stateIdx += 1)
    {
        new (currentStates->LightArray<StateType>::at(stateIdx)) StateType(problem, memory);
        memory += memorySize;
    }

    // Next states
    memorySize = sizeof(Vector<StateType>);
    memory = safeMalloc(memorySize, gpuMallocType);
    Vector<StateType> * const nextStates = new (memory) Vector<StateType>(maxWidth, gpuMallocType);
    memory = StateType::mallocStorages(problem, maxWidth, gpuMallocType);
    memorySize = StateType::sizeOfStorage(problem);
    for(unsigned int stateIdx = 0; stateIdx < currentStates->getCapacity(); stateIdx += 1)
    {
        new (nextStates->LightArray<StateType>::at(stateIdx)) StateType(problem, memory);
        memory += memorySize;
    }

    // Auxiliary information
    memorySize = sizeof(Vector<DP::CostType>);
    memory = safeMalloc(memorySize, gpuMallocType);
    Vector<DP::CostType> * const costs = new (memory) Vector<DP::CostType>(problem->maxBranchingFactor * maxWidth, gpuMallocType);
    memorySize = sizeof(Vector<uint32_t>);
    memory = safeMalloc(memorySize, gpuMallocType);
    Vector<uint32_t> * const indices = new (memory) Vector<uint32_t>(problem->maxBranchingFactor * maxWidth, gpuMallocType);

    // Root
    currentStates->resize(1);
    StateType* const root = currentStates->back();
    DP::makeRoot(problem, root);

    // Build
    unsigned int const variablesCount = problem->variables.getCapacity();
    unsigned int visitedStatesCount = 0;
    uint64_t buildStartTime = now();
    for(unsigned int variableIdx = root->selectedValues.getSize(); variableIdx < variablesCount; variableIdx += 1)
    {
        // Initialize indices
        indices->resize(problem->maxBranchingFactor * currentStates->getSize());
        thrust::sequence(thrust::device, indices->begin(), indices->end());

        // Initialize costs
        costs->resize(problem->maxBranchingFactor * currentStates->getSize());
        thrust::fill(thrust::device, costs->begin(), costs->end(), DP::MaxCost);

        // Calculate costs
        DP::calcCosts<<<currentStates->getSize(), problem->maxBranchingFactor>>>(problem, variableIdx, currentStates, costs);
        cudaDeviceSynchronize();

        // Sort indices by costs
        thrust::sort_by_key(thrust::device, costs->begin(), costs->end(), indices->begin());

        // Discards bad edges by cost
        uint32_t const * const costsEnd = thrust::lower_bound(thrust::device, costs->begin(), costs->end(), DP::MaxCost);
        if (costsEnd != costs->end())
        {
            unsigned int size = costs->indexOf(costsEnd);
            assert(size > 0);
            costs->resize(size);
            indices->resize(size);
        }

        // Adjust next states size
        if (variableIdx < variablesCount - 1)
        {
            if (indices->getSize() > maxWidth)
            {
                printf("[INFO] Max width reached, the solution could be not optimal\n");
                nextStates->resize(maxWidth);
            }
        }
        else
        {
            nextStates->resize(1);
        }
        visitedStatesCount += nextStates->getSize();

        // Add next states
        unsigned int const blockSize = 256;
        unsigned int const blockCount = (nextStates->getSize() / 256) + 1;
        DP::makeStates<<<blockCount, blockSize>>>(problem, variableIdx, currentStates, nextStates, indices, costs);
        cudaDeviceSynchronize();

        printf("[INFO] Time: ");
        printElapsedTime(now() - buildStartTime);
        printf(" | Value: %u", nextStates->at(0)->cost);
        printf(" | Progress: %u/%u", variableIdx + 1, variablesCount);
        printf(" | Visited: %u", visitedStatesCount);
        printf(" | Solution: ");
        nextStates->at(0)->selectedValues.print(true);

        //Prepare for the next loop
        LightVector<StateType>::swap(*currentStates, *nextStates);
    }

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
