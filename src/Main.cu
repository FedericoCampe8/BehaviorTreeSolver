#include <thrust/binary_search.h>
#include <Utils/Chrono.cuh>
#include "DP/VRPModel.cuh"
#include "DD/BuildMetadata.cuh"

using namespace std;
using namespace Memory;
using namespace Chrono;
using namespace DD;
using namespace DP;
using namespace OP;
using ProblemType = VRProblem;
using StateType = VRPState;

// Auxiliary functions
void configGPU();

// Build
__global__ void initBuildMetadata(Vector<BuildMetadata> * buildMetadata);

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
    configGPU();

    // Problem
    ProblemType* const problem = VRProblem::parseGrubHubInstance(problemFileName, MallocType::Managed);

    // Current states
    unsigned int memorySize = sizeof(Vector<StateType>);
    byte* memory = safeMalloc(memorySize, MallocType::Managed);
    Vector<StateType> * const currentStates = new (memory) Vector<StateType>(maxWidth, MallocType::Managed);
    memory = StateType::mallocStorages(problem, maxWidth, MallocType::Managed);
    memorySize = StateType::sizeOfStorage(problem);
    for(unsigned int stateIdx = 0; stateIdx < currentStates->getCapacity(); stateIdx += 1)
    {
        new (currentStates->LightArray<StateType>::at(stateIdx)) StateType(problem, memory);
        memory += memorySize;
    }

    // Next states
    memorySize = sizeof(Vector<StateType>);
    memory = safeMalloc(memorySize, MallocType::Managed);
    Vector<StateType> * const nextStates = new (memory) Vector<StateType>(maxWidth, MallocType::Managed);
    memory = StateType::mallocStorages(problem, maxWidth, MallocType::Managed);
    memorySize = StateType::sizeOfStorage(problem);
    for(unsigned int stateIdx = 0; stateIdx < currentStates->getCapacity(); stateIdx += 1)
    {
        new (nextStates->LightArray<StateType>::at(stateIdx)) StateType(problem, memory);
        memory += memorySize;
    }

    // Build metadata
    memorySize = sizeof(Vector<BuildMetadata>);
    memory = safeMalloc(memorySize, MallocType::Managed);
    Vector<BuildMetadata> * const buildMetadata = new (memory) Vector<BuildMetadata>(problem->maxBranchingFactor * maxWidth, MallocType::Device);
    memorySize = sizeof(BuildMetadata);
    memory = safeMalloc(memorySize, MallocType::Managed);
    BuildMetadata const * const invalidBuildMetadata = new (memory) BuildMetadata();

    // Root
    currentStates->resize(1);
    StateType* const root = currentStates->back();
    DP::makeRoot(problem, root);

    // Build
    unsigned int const variablesCount = problem->variables.getCapacity();
    unsigned int visitedStatesCount = 0;
    uint64_t buildStartTime = now();
    bool printMaxWidthAlert = true;
    for(unsigned int variableIdx = root->selectedValues.getSize(); variableIdx < variablesCount; variableIdx += 1)
    {
        // Initialize build metadata
        buildMetadata->resize(problem->maxBranchingFactor * currentStates->getSize());
        unsigned int const blockSize = 256;
        unsigned int blockCount = (buildMetadata->getSize() / 256) + 1;
        initBuildMetadata<<<blockCount, blockSize>>>(buildMetadata);
        cudaDeviceSynchronize();

        // Calculate costs
        DP::calcCosts<<<currentStates->getSize(), problem->maxBranchingFactor>>>(problem, variableIdx, currentStates, buildMetadata);
        cudaDeviceSynchronize();

        // Sort build metadata
        thrust::sort(thrust::device, buildMetadata->begin(), buildMetadata->end());


        // Discards bad edges by cost
        BuildMetadata const * const buildMetadataEnd = thrust::lower_bound(thrust::device, buildMetadata->begin(), buildMetadata->end(), *invalidBuildMetadata);
        if (buildMetadataEnd != buildMetadata->end())
        {
            unsigned int const size = buildMetadata->indexOf(buildMetadataEnd);
            assert(size > 0);
            buildMetadata->resize(size);
        }

        // Adjust next states size
        if (variableIdx < variablesCount - 1)
        {

            if (buildMetadata->getSize() > maxWidth and printMaxWidthAlert)
            {
                printMaxWidthAlert = false;
                clearLine();
                printf("[INFO] Max width reached");
                printf(" | Time: ");
                printElapsedTime(now() - buildStartTime);
                printf(" | Progress: %u/%u", variableIdx + 1, variablesCount);
                printf(" | Visited: %u\n", visitedStatesCount);
            }
            unsigned int const size = min(maxWidth, buildMetadata->getSize());
            nextStates->resize(size);
        }
        else
        {
            nextStates->resize(1);
        }
        visitedStatesCount += nextStates->getSize();

        // Add next states
        blockCount = (nextStates->getSize() / 256) + 1;
        DP::makeStates<<<blockCount, blockSize>>>(problem, variableIdx, currentStates, nextStates, buildMetadata);
        cudaDeviceSynchronize();

        clearLine();
        printf("[INFO] Time: ");
        printElapsedTime(now() - buildStartTime);
        printf(" | Value: %u", nextStates->at(0)->cost);
        printf(" | Progress: %u/%u", variableIdx + 1, variablesCount);
        printf(" | Visited: %u", visitedStatesCount);
        printf(" | Solution: ");
        nextStates->at(0)->selectedValues.print(false);
        printf("%s", variableIdx + 1 < variablesCount ? "\r" : "\n");
        fflush(stdout);

        // Prepare for the next loop
        LightVector<StateType>::swap(*currentStates, *nextStates);
    }

    return EXIT_SUCCESS;
}

void configGPU()
{
    cudaDeviceReset();

    //Heap
    std::size_t sizeHeap = 100ul * 1024ul * 1024ul;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, sizeHeap);

    //Stack
    size_t sizeStackThread = 4 * 1024;
    cudaDeviceSetLimit(cudaLimitStackSize, sizeStackThread);

    //Cache
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
}


__global__
void initBuildMetadata(Vector<BuildMetadata> * buildMetadata)
{
    unsigned int buildMetadataIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if(buildMetadataIdx < buildMetadata->getSize())
    {
        new (buildMetadata->at(buildMetadataIdx)) BuildMetadata(DP::MaxCost, buildMetadataIdx);
    }
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
