#include <algorithm>
#include <thread>
#include <External/AnyOption/anyoption.h>
#include <Utils/Algorithms.cuh>
#include <Utils/Chrono.cuh>
#include <LNS/Context.h>
#include "DP/TSPPDModel.cuh"
#include "DP/CTWPModel.cuh"
#include "DP/MOSPModel.cuh"
#include "DP/JSPModel.cuh"
#include "DP/SOPModel.cuh"
#include "LNS/OffloadBufferCPU.cuh"
#include "LNS/OffloadBufferGPU.cuh"
#include "LNS/StatesPriorityQueue.cuh"
#include "Options.h"

using namespace std;
using namespace Memory;
using namespace Chrono;
using namespace DD;
using namespace DP;
using namespace OP;
using namespace LNS;
using ProblemType = TSPPDProblem;
using StateType = TSPPDState;

// Auxiliary functions
AnyOption* parseOptions(int argc, char* argv[]);

void configGPU();

// Debug
void printElapsedTime(u64 elapsedTimeMs);

void clearLine();

int main(int argc, char* argv[])
{
    u64 startTime = now();
    
    // Options parsing
    Options options;
    if (not options.parseOptions(argc, argv))
    {
        return EXIT_FAILURE;
    }
    else
    {
        options.printOptions();
    }

    // Context initialization
    MallocType mallocTypeGpu = MallocType::Std;
    if (options.parallelismGpu > 0)
    {
        configGPU();
        mallocTypeGpu = MallocType::Managed;
    };

    //Problem
    ProblemType* const problemCpu = parseInstance<ProblemType>(options.inputFilename, MallocType::Std);
    ProblemType* const problemGpu = parseInstance<ProblemType>(options.inputFilename, mallocTypeGpu);

    // StatesPriorityQueue
    StatesPriorityQueue<StateType> statesPriorityQueue(problemCpu, options.queueSize);

    // OffloadBuffers
    byte* memory = nullptr;
    memory = safeMalloc(sizeof(OffloadBufferCPU<ProblemType,StateType>), MallocType::Std);
    OffloadBufferCPU<ProblemType,StateType>* offloadBufferCpu = new (memory) OffloadBufferCPU<ProblemType, StateType>(problemCpu, options);
    memory = safeMalloc(sizeof(OffloadBufferGPU<ProblemType,StateType>), mallocTypeGpu);
    OffloadBufferGPU<ProblemType,StateType>* offloadBufferGpu = new (memory) OffloadBufferGPU<ProblemType,StateType>(problemGpu, options, mallocTypeGpu);

    // Solutions
    memory = safeMalloc(sizeof(StateType), mallocTypeGpu);
    StateType* bestSolution = new (memory) StateType(problemCpu, mallocTypeGpu);
    bestSolution->makeInvalid();
    memory = safeMalloc(sizeof(StateType), mallocTypeGpu);
    StateType* currentSolution = new (memory) StateType(problemCpu, mallocTypeGpu);
    currentSolution->makeInvalid();

    // Root
    memory = safeMalloc(sizeof(StateType), mallocTypeGpu);
    StateType* root = new (memory) StateType(problemCpu, mallocTypeGpu);
    makeRoot(problemCpu, root);
    statesPriorityQueue.insert(root);

    // LNS
    u32 iterationsCount = 0;
    SearchPhase searchPhase = SearchPhase::Init;

    clearLine();
    u64 searchStartTime = now();
    printf("[INFO] Start initial search");
    printf(" | Time: ");
    printElapsedTime(now() - startTime);
    printf("\n");
    do
    {

        u64 iterationStartTime = now();

        switch(searchPhase)
        {
            case SearchPhase::Init:
            {
                if (statesPriorityQueue.isFull())
                {
                    searchPhase = SearchPhase::LNS;
                    clearLine();
                    printf("[INFO] Switch to LNS");
                    printf(" | Time: ");
                    printElapsedTime(now() - startTime);
                    printf("\n");

                    offloadBufferCpu->initializeRngsAsync(options.randomSeed);
                    offloadBufferGpu->initializeRngsAsync(options.randomSeed);

                    offloadBufferCpu->wait();
                    offloadBufferGpu->wait();

                    *currentSolution = *bestSolution;

                    continue;
                }

                offloadBufferCpu->initializeOffload(&statesPriorityQueue);
                offloadBufferGpu->initializeOffload(&statesPriorityQueue);
            }
            break;

            case SearchPhase::LNS:
            {
                offloadBufferCpu->generateNeighbourhoodsAsync(bestSolution);
                offloadBufferGpu->generateNeighbourhoodsAsync(bestSolution);

                offloadBufferCpu->wait();
                offloadBufferGpu->wait();

                offloadBufferCpu->initializeOffload(root);
                offloadBufferGpu->initializeOffload(root);

                currentSolution->makeInvalid();
            }
            break;
        }

        u64 offloadStartTime = now();

        offloadBufferCpu->doOffloadAsync(searchPhase);
        offloadBufferGpu->doOffloadAsync(searchPhase);

        offloadBufferCpu->wait();
        offloadBufferGpu->wait();

        u64 offloadElapsedTime = max(now() - offloadStartTime, 1ul);

        if(searchPhase ==  SearchPhase::Init)
        {
            offloadBufferCpu->finalizeOffload(&statesPriorityQueue);
            offloadBufferGpu->finalizeOffload(&statesPriorityQueue);
        }

        u64 iterationEndTime = now();

        bool betterSolutionFromCpu = false;
        StateType const * bestSolutionCpu = offloadBufferCpu->getBestSolution(searchPhase);
        if(bestSolutionCpu != nullptr)
        {
            if(*bestSolutionCpu < *currentSolution)
            {
                *currentSolution = *bestSolutionCpu;
            }

            if(*bestSolutionCpu < *bestSolution)
            {
                betterSolutionFromCpu = true;
                *bestSolution = *bestSolutionCpu;
            }
        }

        bool betterSolutionFromGpu = false;
        StateType const * bestSolutionGpu = offloadBufferGpu->getBestSolution(searchPhase);
        if(bestSolutionGpu != nullptr)
        {
            if(*bestSolutionGpu < *currentSolution)
            {
                *currentSolution = *bestSolutionGpu;
            }

            if(*bestSolutionGpu < *bestSolution)
            {
                betterSolutionFromGpu = true;
                *bestSolution = *bestSolutionGpu;
            }
        }

        if(betterSolutionFromCpu or betterSolutionFromGpu)
        {
            clearLine();
            printf("[SOLUTION] Source: %s", betterSolutionFromCpu ? "CPU" : "GPU");
            printf(" | Time: ");
            printElapsedTime(now() - searchStartTime);
            printf(" | Iteration: %u ", iterationsCount);
            printf(" | Cost: %u", bestSolution->cost);
            printf(" | Solution: ");
            bestSolution->print(true);
        }
        else
        {
            if(options.statistics)
            {
                u64 cpuSpeed = 0;
                if (offloadBufferCpu->getSize() > 0)
                {
                    cpuSpeed = options.parallelismCpu * 1000 / offloadElapsedTime;
                }

               u64 gpuSpeed = 0;
                if (offloadBufferGpu->getSize() > 0)
                {;
                    gpuSpeed = options.parallelismGpu * 1000 / offloadElapsedTime;
                }

                printf("[INFO] Current value: %u", currentSolution->cost);
                printf(" | Time: ");
                printElapsedTime(now() - searchStartTime);
                printf(" | Iteration: %u ", iterationsCount);
                printf("(");
                printElapsedTime(iterationEndTime - iterationStartTime);
                printf(") ");
                printf(" | CPU: %lu MDD/s | GPU: %lu MDD/s\r", cpuSpeed, gpuSpeed);
            }
        }
        fflush(stdout);
        iterationsCount += 1;
    }
    while(now() - searchStartTime < options.timeout * 1000 and (not statesPriorityQueue.isEmpty()));


    return EXIT_SUCCESS;
}

void configGPU()
{
    //Heap
    std::size_t const sizeHeap = 4ul * 1024ul * 1024ul * 1024ul;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, sizeHeap);

    //Stack
    size_t const sizeStackThread = 4 * 1024;
    cudaDeviceSetLimit(cudaLimitStackSize, sizeStackThread);

    //Cache
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
}


void printElapsedTime(u64 elapsedTimeMs)
{
    u64 ms = elapsedTimeMs;

    u64 const h = ms / (1000 * 60 * 60);
    ms -= h * 1000 * 60 * 60;

    u64 const m = ms / (1000 * 60);
    ms -= m * 1000 * 60;
    
    u64 const s = ms / 1000;
    ms -= s * 1000;

    printf("%02lu:%02lu:%02lu.%03lu", h, m, s, ms);
}

void clearLine()
{
    // ANSI clear line escape code
    printf("\33[2K\r");
}