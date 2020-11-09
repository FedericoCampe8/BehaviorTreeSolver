#include <cstdio>
#include <cstddef>
#include <new>
#include <utility>

#include <Utils/Memory.cuh>
#include <Utils/Chrono.cuh>
#include <Utils/CUDA.cuh>

#include "OP/TSPProblem.cuh"
#include "MDD/MDD.cuh"
#include "DP/TSPModel.cuh"

using namespace std;

//Auxiliary functions
void initProblem(OP::TSPProblem* problem);
__global__ void initRoots(MDD::MDD::Type type, unsigned int width, unsigned int * rootsCount, DP::TSPState * roots, unsigned int * levels, OP::TSPProblem const * problem);
__global__ void buildMDDs(MDD::MDD::Type type, unsigned int width, DP::TSPState const * roots, unsigned int const * levels, unsigned int* printLock, OP::TSPProblem const * problem);

int main(int argc, char ** argv)
{
    // Heap
    std::size_t sizeHeap = 3ul * 1024ul * 1024ul * 1024ul;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, sizeHeap);

    // Stack
    size_t sizeStackThread = 4 * 1024;
    cudaDeviceSetLimit(cudaLimitStackSize, sizeStackThread);

    // Cache
    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );

    //Problem
    unsigned int varsCount = 6;
    std::size_t problemSize = sizeof(OP::TSPProblem);
    std::size_t problemStorageSize = OP::TSPProblem::sizeofStorage(varsCount);
    std::byte* mem = Memory::safeManagedMalloc(problemSize + problemStorageSize);
    OP::TSPProblem* problem = reinterpret_cast<OP::TSPProblem*>(mem);
    byte* problemStorage = mem + problemSize;
    new (problem) OP::TSPProblem(varsCount, problemStorage);
    initProblem(problem);

    //Roots and Levels
    unsigned int width = std::atoi(argv[1]);
    unsigned int* rootsCount = reinterpret_cast<unsigned int*>(Memory::safeManagedMalloc((sizeof(unsigned int))));
    *rootsCount = (problem->vars.size * + 1) * width;
    unsigned int rootSize = sizeof(DP::TSPState);
    unsigned int rootStorageSize = DP::TSPState::sizeofStorage(problem->vars.size);
    DP::TSPState* roots =  reinterpret_cast<DP::TSPState*>(Memory::safeManagedMalloc(rootSize * *rootsCount));
    std::byte* rootsStorageMem = Memory::safeManagedMalloc(rootStorageSize * *rootsCount);
    unsigned int* levels =  reinterpret_cast<unsigned int*>(Memory::safeManagedMalloc(sizeof(unsigned int) * *rootsCount));
    for(unsigned int rootIdx = 0; rootIdx < *rootsCount; rootIdx += 1)
    {
        new (&roots[rootIdx]) DP::TSPState(problem->vars.size, &rootsStorageMem[rootStorageSize * rootIdx]);
        levels[rootIdx] = 0;
    }
    DP::TSPModel::makeRoot(problem, &roots[0]);
    initRoots<<<1,32>>>(MDD::MDD::Type::Relaxed, width, rootsCount, roots, levels, problem);
    cudaDeviceSynchronize();

    //MDD
    unsigned int* printLock = reinterpret_cast<unsigned int*>(Memory::safeManagedMalloc((sizeof(unsigned int))));
    *printLock = 0;
    auto start = Chrono::now();
    buildMDDs<<<*rootsCount,1>>>(MDD::MDD::Type::Relaxed, width, roots, levels, printLock, problem);
    cudaDeviceSynchronize();
    auto end = Chrono::now();
    printf("[INFO] Created %d MDDs in %d ms\n", *rootsCount, static_cast<unsigned int>(end - start));
    return EXIT_SUCCESS;
}

void initProblem(OP::TSPProblem* problem)
{
    new (&problem->vars[0]) OP::Variable(0,0);
    new (&problem->vars[1]) OP::Variable(2,5);
    new (&problem->vars[2]) OP::Variable(2,5);
    new (&problem->vars[3]) OP::Variable(2,5);
    new (&problem->vars[4]) OP::Variable(2,5);
    new (&problem->vars[5]) OP::Variable(1,1);

    problem->setStartEndLocations(0,1);

    problem->addPickupDelivery(2,3);
    problem->addPickupDelivery(4,5);

    unsigned int costs[36] = {
        0,0,389,792,1357,961,
        0,0,0,0,0,0,
        389,0,0,641,1226,1168,
        792,0,641,0,1443,1490,
        1357,0,1226,1443,0,741,
        961,0,1168,1490,741,0
    };

    for(unsigned int i = 0; i < 36; i+= 1)
    {
      problem->distances[i] = costs[i];
    }
}

__global__
void initRoots(MDD::MDD::Type type, unsigned int width, unsigned int* rootsCount, DP::TSPState* roots, unsigned int* levels, OP::TSPProblem const * problem)
{
    __shared__ unsigned int alignedSharedMem[1000];
    std::byte* sharedMem = reinterpret_cast<std::byte*>(alignedSharedMem);

    ONE_THREAD_IN_BLOCK
    {
        //MDD
        auto start = Chrono::now();
        MDD::MDD mdd(type, width, roots, 0, problem);
        mdd.buildTopDown(sharedMem);
        auto end = Chrono::now();
        printf("[INFO] Master MDD created in %d ms\n", static_cast<unsigned int>(end - start));

        //Copy states for next iteration
        start = Chrono::now();
        unsigned int rootIdx  = 0;
        for (unsigned int level = 0; level < mdd.dag.height; level += 1)
        {
            DP::TSPState* states = mdd.dag.getStates(level);
            for(unsigned int stateIdx = 0; stateIdx < mdd.dag.width; stateIdx += 1)
            {
                DP::TSPState const & state = states[stateIdx];
                if (DP::TSPState::isActive(state))
                {
                    roots[rootIdx] = state;
                    levels[rootIdx] = level;
                    rootIdx += 1;
                }
            }
        }
        end = Chrono::now();
        printf("[INFO] Roots initialized in %d ms\n", static_cast<unsigned int>(end - start));
        *rootsCount = rootIdx;
    }
}

__global__
void buildMDDs(MDD::MDD::Type type, unsigned int width, DP::TSPState const * roots, unsigned int const * levels, unsigned int* printLock, OP::TSPProblem const * problem)
{
    __shared__ unsigned int alignedSharedMem[1000];
    std::byte* sharedMem = reinterpret_cast<std::byte*>(alignedSharedMem);

    ONE_THREAD_IN_BLOCK
    {
        //MDD
        auto start = Chrono::now();
        MDD::MDD mdd(type, width, roots + blockIdx.x, levels[blockIdx.x], problem);
        auto end = Chrono::now();
        //printf("[INFO] MDD %d created in %d ms\n", blockIdx.x, static_cast<unsigned int>(end - start));


        start = Chrono::now();
        mdd.buildTopDown(sharedMem);
        end = Chrono::now();
        printf("[INFO] MDD %d built in %d ms\n", blockIdx.x, static_cast<uint>(end - start));

        /*
        startLocation = Chrono::now();
        StaticVector<int> labels(mdd->height);
        mdd->DFS(0, 0,labels, true);
        end = Chrono::now();
        printf("[INFO] DFS performed in %d ms\n", static_cast<uint>(end - start));
        */
    }
}

