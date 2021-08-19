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
#include "LNS/SearchManagerCPU.cuh"
#include "LNS/SearchManagerGPU.cuh"
#include "LNS/StatesPriorityQueue.cuh"
#include "LNS/SyncState.cuh"
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
template<typename ProblemType, typename StateType>
void printStatistics(u64 startTime, SearchManagerCPU<ProblemType,StateType>* searchManagerCpu, SearchManagerGPU<ProblemType,StateType>* searchManagerGpu);

template<typename ProblemType, typename StateType>
void printBestSolution(StateType* bestSolution, u64 startTime, SearchManagerCPU<ProblemType,StateType>* searchManagerCpu, SearchManagerGPU<ProblemType,StateType>* searchManagerGpu);

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
    if (options.mddsGpu > 0)
    {
        configGPU();
        mallocTypeGpu = MallocType::Managed;
    };

    //Problem
    ProblemType* const problemCpu = parseInstance<ProblemType>(options.inputFilename, MallocType::Std);
    ProblemType* const problemGpu = parseInstance<ProblemType>(options.inputFilename, mallocTypeGpu);

    // Queue
    StatesPriorityQueue<StateType> statesPriorityQueue(problemCpu, options.queueSize);

    // Search managers
    byte* memory = nullptr;
    memory = safeMalloc(sizeof(SearchManagerCPU<ProblemType,StateType>), MallocType::Std);
    SearchManagerCPU<ProblemType,StateType>* searchManagerCpu = new (memory) SearchManagerCPU<ProblemType, StateType>(problemCpu, &options);
    memory = safeMalloc(sizeof(SearchManagerGPU<ProblemType,StateType>), mallocTypeGpu);
    SearchManagerGPU<ProblemType,StateType>* searchManagerGpu = new (memory) SearchManagerGPU<ProblemType, StateType>(problemGpu, &options, mallocTypeGpu);

    // Solution
    memory = safeMalloc(sizeof(StateType), mallocTypeGpu);
    StateType* bestSolution = new (memory) StateType(problemCpu, mallocTypeGpu);
    bestSolution->invalidate();

    // Root
    memory = safeMalloc(sizeof(StateType), mallocTypeGpu);
    StateType* root = new (memory) StateType(problemCpu, mallocTypeGpu);
    makeRoot(problemCpu, root);
    statesPriorityQueue.insert(root);

    // Search
    bool timeout = false;

    clearLine();
    u64 searchStartTime = now();
    printf("[INFO] Start initial search");
    printf(" | Time: ");
    printElapsedTime(now() - startTime);
    printf("\n");

    std::thread searchInitCpu(&SearchManagerCPU<ProblemType,StateType>::searchInitLoop, searchManagerCpu, &statesPriorityQueue, &timeout);
    searchInitCpu.detach();
    std::thread searchInitGpu(&SearchManagerGPU<ProblemType,StateType>::searchInitLoop, searchManagerGpu, &statesPriorityQueue, &timeout);
    searchInitGpu.detach();

    while(not timeout)
    {
        if (searchManagerCpu->done and searchManagerGpu->done)
        {
            break;
        }

        if (options.statistics)
        {
            printStatistics(startTime, searchManagerCpu, searchManagerGpu);
        }

        printBestSolution(bestSolution, startTime, searchManagerCpu, searchManagerGpu);

        std::this_thread::sleep_for(std::chrono::seconds(1));
        timeout = (now() - searchStartTime) < (options.timeout * 1000);
    }

    searchManagerCpu->initializeRngs(&timeout);
    searchManagerGpu->initializeRngs(&timeout);

    std::thread searchLnsCpu(&SearchManagerCPU<ProblemType,StateType>::searchLnsLoop, searchManagerCpu, &timeout);
    searchLnsCpu.detach();
    std::thread searchLnsGpu(&SearchManagerGPU<ProblemType,StateType>::searchLnsLoop, searchManagerGpu, &timeout);
    searchInitGpu.detach();

    while(not timeout)
    {
        if (searchManagerCpu->done and searchManagerGpu->done)
        {
            break;
        }

        if (options.statistics)
        {
            printStatistics(startTime, searchManagerCpu, searchManagerGpu);
        }

        printBestSolution(bestSolution, startTime, searchManagerCpu, searchManagerGpu);

        std::this_thread::sleep_for(std::chrono::seconds(1));
        timeout = (now() - searchStartTime) < (options.timeout * 1000);
    }

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

template<typename ProblemType, typename StateType>
void printStatistics(u64 startTime, SearchManagerCPU<ProblemType,StateType>* searchManagerCpu, SearchManagerGPU<ProblemType,StateType>* searchManagerGPU)
{
    printf("[INFO] Time: ");
    printElapsedTime(now() - startTime);
    printf(" | CPU: %lu MDD/s | GPU: %lu MDD/s\r", searchManagerCpu->speed, searchManagerGPU->speed);
}

template<typename ProblemType, typename StateType>
void printBestSolution(StateType* bestSolution, u64 startTime, SearchManagerCPU<ProblemType, StateType>* searchManagerCpu, SearchManagerGPU<ProblemType, StateType>* searchManagerGpu)
{
    bool betterSolutionFromCpu = false;
    bool betterSolutionFromGpu = false;

    searchManagerCpu->bestSolution.mutex.lock();
    searchManagerCpu->neighborhoodSolution.mutex.lock();
    searchManagerGpu->bestSolution.mutex.lock();
    searchManagerGpu->neighborhoodSolution.mutex.lock();

    if(searchManagerCpu->bestSolution.state.cost < bestSolution->cost)
    {
        *bestSolution = searchManagerCpu->bestSolution.state;
        searchManagerGpu->neighborhoodSolution.state = searchManagerCpu->bestSolution.state;
        betterSolutionFromCpu = true;
    }

    if(searchManagerGpu->bestSolution.state.cost < bestSolution->cost)
    {
        *bestSolution = searchManagerGpu->bestSolution.state;
        searchManagerCpu->neighborhoodSolution.state = searchManagerGpu->bestSolution.state;
        betterSolutionFromGpu = true;
    }

    if (betterSolutionFromCpu or betterSolutionFromGpu)
    {
        *bestSolution = searchManagerCpu->bestSolution.state;
        clearLine();
        printf("[SOLUTION] Source: %s", betterSolutionFromCpu ? "CPU" : "GPU");
        printf(" | Time: ");
        printElapsedTime(now() - startTime);
        printf(" | Cost: %u", bestSolution->cost);
        printf(" | Solution: ");
        bestSolution->print(true);
    }

    searchManagerCpu->bestSolution.mutex.unlock();
    searchManagerCpu->neighborhoodSolution.mutex.unlock();
    searchManagerGpu->bestSolution.mutex.unlock();
    searchManagerGpu->neighborhoodSolution.mutex.unlock();
}
