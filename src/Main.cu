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
#include <thrust/equal.h>

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
void printStatistics(Options const * options, u64 startTime, SearchManagerCPU<ProblemType,StateType>* searchManagerCpu, SearchManagerGPU<ProblemType,StateType>* searchManagerGpu);

template<typename ProblemType, typename StateType>
void checkBetterSolutions(StateType* bestSolution, u64 startTime, SearchManagerCPU<ProblemType, StateType>* searchManagerCpu, SearchManagerGPU<ProblemType, StateType>* searchManagerGpu);

bool checkTimeout(u64 startTime, Options const * options);

// Debug
void printElapsedTime(u64 elapsedTimeMs);

void clearLine();

int main(int argc, char* argv[])
{
    u64 startTime = now();
    
    // Options parsing
    byte* memory = nullptr;
    memory = safeMalloc(sizeof(Options), MallocType::Std);
    Options* optionsCpu = new (memory) Options();
    if (not optionsCpu->parseOptions(argc, argv))
    {
        return EXIT_FAILURE;
    }
    else
    {
        optionsCpu->printOptions();
    }

    // Context initialization
    MallocType mallocTypeGpu = MallocType::Std;
    if (optionsCpu->mddsGpu > 0)
    {
        configGPU();
        mallocTypeGpu = MallocType::Managed;
    };

    memory = safeMalloc(sizeof(Options), mallocTypeGpu);
    Options* optionsGpu = new (memory) Options();
    optionsGpu->parseOptions(argc, argv);

    //Problem
    ProblemType* const problemCpu = parseInstance<ProblemType>(optionsCpu->inputFilename, MallocType::Std);
    ProblemType* const problemGpu = parseInstance<ProblemType>(optionsGpu->inputFilename, mallocTypeGpu);

    // Queue
    StatesPriorityQueue<StateType> statesPriorityQueue(problemCpu, optionsCpu->queueSize);

    // Search managers
    memory = nullptr;
    memory = safeMalloc(sizeof(SearchManagerCPU<ProblemType,StateType>), MallocType::Std);
    SearchManagerCPU<ProblemType,StateType>* searchManagerCpu = new (memory) SearchManagerCPU<ProblemType, StateType>(problemCpu, optionsCpu);
    memory = safeMalloc(sizeof(SearchManagerGPU<ProblemType,StateType>), mallocTypeGpu);
    SearchManagerGPU<ProblemType,StateType>* searchManagerGpu = new (memory) SearchManagerGPU<ProblemType, StateType>(problemGpu, optionsGpu, mallocTypeGpu);

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

    while(not (timeout or (searchManagerCpu->done and searchManagerGpu->done)))
    {
        printStatistics(optionsCpu, startTime, searchManagerCpu, searchManagerGpu);
        checkBetterSolutions(bestSolution, startTime, searchManagerCpu, searchManagerGpu);
        std::this_thread::sleep_for(std::chrono::seconds(1));
        timeout = checkTimeout(startTime, optionsCpu);
    }
    checkBetterSolutions(bestSolution, startTime, searchManagerCpu, searchManagerGpu);

    clearLine();
    printf("[INFO] Switch to LNS");
    printf(" | Time: ");
    printElapsedTime(now() - startTime);
    printf("\n");

    searchManagerCpu->initializeRngs(&timeout);
    searchManagerGpu->initializeRngs(&timeout);

    std::thread searchLnsCpu(&SearchManagerCPU<ProblemType,StateType>::searchLnsLoop, searchManagerCpu, root, &timeout);
    searchLnsCpu.detach();
    std::thread searchLnsGpu(&SearchManagerGPU<ProblemType,StateType>::searchLnsLoop, searchManagerGpu, root, &timeout);
    searchLnsGpu.detach();

    while (not timeout)
    {
        printStatistics(optionsCpu, startTime, searchManagerCpu, searchManagerGpu);
        checkBetterSolutions(bestSolution, startTime, searchManagerCpu, searchManagerGpu);
        std::this_thread::sleep_for(std::chrono::seconds(1));
        timeout = checkTimeout(startTime, optionsCpu);
    }
    checkBetterSolutions(bestSolution, startTime, searchManagerCpu, searchManagerGpu);


    if(optionsCpu->mddsGpu > 0)
    {
        cudaDeviceReset();
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
void printStatistics(Options const * options, u64 startTime, SearchManagerCPU<ProblemType,StateType>* searchManagerCpu, SearchManagerGPU<ProblemType,StateType>* searchManagerGpu)
{
    if(options->statistics)
    {
        clearLine();
        printf("[INFO] Time: ");
        printElapsedTime(now() - startTime);

        if(options->mddsCpu > 0)
        {
            searchManagerCpu->bestSolution.mutex.lock();
            DP::CostType currentCostCpu = searchManagerCpu->bestSolution.state.cost;
            searchManagerCpu->bestSolution.mutex.unlock();
            printf(" | CPU: %d - %lu - %lu MDD/s",  currentCostCpu , searchManagerCpu->iteration, searchManagerCpu->speed);
        }

        if (options->mddsGpu > 0)
        {
            searchManagerGpu->bestSolution.mutex.lock();
            DP::CostType currentCostGpu = searchManagerGpu->bestSolution.state.cost;
            searchManagerGpu->bestSolution.mutex.unlock();
            printf(" | GPU: %d - %lu - %lu MDD/s", currentCostGpu, searchManagerGpu->iteration, searchManagerGpu->speed);
        }

        printf("\r");
        fflush(stdout);
    }
}

template<typename ProblemType, typename StateType>
void checkBetterSolutions(StateType* bestSolution, u64 startTime, SearchManagerCPU<ProblemType, StateType>* searchManagerCpu, SearchManagerGPU<ProblemType, StateType>* searchManagerGpu)
{
    bool betterSolutionFromCpu = false;
    bool betterSolutionFromGpu = false;

    searchManagerCpu->bestSolution.mutex.lock();
    if(searchManagerCpu->bestSolution.state.cost < bestSolution->cost)
    {
        *bestSolution = searchManagerCpu->bestSolution.state;
        betterSolutionFromCpu = true;
    }
    searchManagerCpu->bestSolution.mutex.unlock();

    searchManagerGpu->bestSolution.mutex.lock();
    if(searchManagerGpu->bestSolution.state.cost < bestSolution->cost)
    {
        *bestSolution = searchManagerGpu->bestSolution.state;
        betterSolutionFromGpu = true;
    }
    searchManagerGpu->bestSolution.mutex.unlock();

    if(betterSolutionFromCpu or betterSolutionFromGpu)
    {
        searchManagerCpu->neighborhoodSolution.mutex.lock();
        searchManagerCpu->neighborhoodSolution.state = *bestSolution;
        searchManagerCpu->neighborhoodSolution.mutex.unlock();

        searchManagerGpu->neighborhoodSolution.mutex.lock();
        searchManagerGpu->neighborhoodSolution.state = *bestSolution;
        searchManagerGpu->neighborhoodSolution.mutex.unlock();
    }

    if (betterSolutionFromCpu or betterSolutionFromGpu)
    {
        clearLine();
        printf("[SOLUTION] Source: %s", betterSolutionFromCpu ? "CPU" : "GPU");
        printf(" | Time: ");
        printElapsedTime(now() - startTime);
        printf(" | Cost: %u", bestSolution->cost);
        printf(" | Solution: ");
        bestSolution->print(true);
    }
    fflush(stdout);
}

bool checkTimeout(u64 startTime, Options const* options)
{
    return  (now() - startTime) > (options->timeout * 1000);
}
