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
__global__ void initRoots(MDD::MDD::Type type, unsigned int width, unsigned int * rootsCount, DP::TSPState * roots, unsigned int * levels, OP::TSPProblem const * problem, unsigned int cutsetSize,  DP::TSPState * cutsetStates);
__global__ void buildMDDs(unsigned int width, DP::TSPState const * roots, unsigned int const * levels, unsigned int* bounds, OP::TSPProblem const * problem, unsigned int cutsetSize, DP::TSPState * cutsetStates);

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
    unsigned int varsCount = 8;
    std::size_t problemSize = sizeof(OP::TSPProblem);
    std::size_t problemStorageSize = OP::TSPProblem::sizeofStorage(varsCount);
    std::byte* mem = Memory::safeManagedMalloc(problemSize + problemStorageSize);
    OP::TSPProblem* problem = reinterpret_cast<OP::TSPProblem*>(mem);
    byte* problemStorage = mem + problemSize;
    new (problem) OP::TSPProblem(varsCount, problemStorage);
    initProblem(problem);

    //Roots and Levels
    unsigned int width = std::atoi(argv[1]);
    unsigned int* rootsCount = reinterpret_cast<unsigned int*>(Memory::safeManagedMalloc(sizeof(unsigned int)));
    *rootsCount = (problem->vars.size * + 1) * width;
    unsigned int stateSize = sizeof(DP::TSPState);
    unsigned int stateStorageSize = DP::TSPState::sizeofStorage(problem->vars.size);
    DP::TSPState* roots =  reinterpret_cast<DP::TSPState*>(Memory::safeManagedMalloc(stateSize * *rootsCount));
    std::byte* rootsStorageMem = Memory::safeManagedMalloc(stateStorageSize * *rootsCount);
    unsigned int* levels =  reinterpret_cast<unsigned int*>(Memory::safeManagedMalloc(sizeof(unsigned int) * *rootsCount));
    for(unsigned int rootIdx = 0; rootIdx < *rootsCount; rootIdx += 1)
    {
        new (&roots[rootIdx]) DP::TSPState(problem->vars.size, &rootsStorageMem[stateStorageSize * rootIdx]);
        levels[rootIdx] = 0;
    }

    //Cutset
    unsigned int cutsetSize = width * MDD::MDD::calcFanout(problem, 0);
    unsigned int cutsetsStatesCount = *rootsCount * cutsetSize;
    DP::TSPState* cutsetsStates =  reinterpret_cast<DP::TSPState*>(Memory::safeManagedMalloc(stateSize * cutsetsStatesCount));
    std::byte* cutsetsStateStorageMem = Memory::safeManagedMalloc(stateStorageSize * cutsetsStatesCount);
    for(unsigned int stateIdx = 0; stateIdx < cutsetsStatesCount; stateIdx += 1)
    {
        new (&cutsetsStates[stateIdx]) DP::TSPState(problem->vars.size, &cutsetsStateStorageMem[stateStorageSize * stateIdx]);
    }

    DP::TSPModel::makeRoot(problem, &roots[0]);
    initRoots<<<1,32>>>(MDD::MDD::Type::Relaxed, width, rootsCount, roots, levels, problem, cutsetSize, cutsetsStates);
    cudaDeviceSynchronize();


    //Bounds
    unsigned int* bounds = reinterpret_cast<unsigned int*>(Memory::safeManagedMalloc(2 * *rootsCount * sizeof(unsigned int)));

    //MDD
    auto start = Chrono::now();
    buildMDDs<<<*rootsCount,1>>>(width, roots, levels, bounds, problem, cutsetSize, cutsetsStates);
    cudaDeviceSynchronize();
    auto end = Chrono::now();
    printf("[INFO] Created %d MDDs in %d ms\n", *rootsCount, static_cast<unsigned int>(end - start));

    return EXIT_SUCCESS;
}

void initProblem(OP::TSPProblem* problem)
{

    /* grubhub-02-0.json
    new (&problem->vars[0]) OP::Variable(0,0);
    new (&problem->vars[1]) OP::Variable(2,5);
    new (&problem->vars[2]) OP::Variable(2,5);
    new (&problem->vars[3]) OP::Variable(2,5);
    new (&problem->vars[4]) OP::Variable(2,5);
    new (&problem->vars[5]) OP::Variable(1,1);

    problem->setStartEndLocations(0,1);

    problem->addPickupDelivery(2,3);
    problem->addPickupDelivery(4,5);

    unsigned int costs[6*6] = {
        0,0,389,792,1357,961,
        0,0,0,0,0,0,
        389,0,0,641,1226,1168,
        792,0,641,0,1443,1490,
        1357,0,1226,1443,0,741,
        961,0,1168,1490,741,0
    };

    for(unsigned int i = 0; i < 6*6; i+= 1)
    {
      problem->distances[i] = costs[i];
    }
     */

    /* grubhub-03-0.json*/
   new (&problem->vars[0]) OP::Variable(0,0);
   new (&problem->vars[1]) OP::Variable(2,7);
   new (&problem->vars[2]) OP::Variable(2,7);
   new (&problem->vars[3]) OP::Variable(2,7);
   new (&problem->vars[4]) OP::Variable(2,7);
   new (&problem->vars[5]) OP::Variable(2,7);
   new (&problem->vars[6]) OP::Variable(2,7);
   new (&problem->vars[7]) OP::Variable(1,1);

   problem->setStartEndLocations(0,1);

   problem->addPickupDelivery(2,3);
   problem->addPickupDelivery(4,5);
   problem->addPickupDelivery(6,7);

   unsigned int costs[8*8] = {
       0,    0, 1270,  906, 1024,  726, 1206, 1262,
       0,    0,    0,    0,    0,    0,    0,    0,
       1270,    0,    0,  518,  327,  962,  219,  505,
       906,    0,  518,    0,  349,  501,  400,  369,
       1024,    0,  327,  349,    0,  727,  184,  525,
       726,    0,  962,  501,  727,    0,  783,  907,
       1206,    0,  219,  400,  184,  783,    0,  591,
       1262,    0,  505,  369,  525,  907,  591,    0
   };

   for(unsigned int i = 0; i < 8*8; i+= 1)
   {
     problem->distances[i] = costs[i];
   }

    /* grubhub-05-0.json
   new (&problem->vars[0]) OP::Variable(0,0);
   new (&problem->vars[1]) OP::Variable(2,11);
   new (&problem->vars[2]) OP::Variable(2,11);
   new (&problem->vars[3]) OP::Variable(2,11);
   new (&problem->vars[4]) OP::Variable(2,11);
   new (&problem->vars[5]) OP::Variable(2,11);
   new (&problem->vars[6]) OP::Variable(2,11);
   new (&problem->vars[7]) OP::Variable(2,11);
   new (&problem->vars[8]) OP::Variable(2,11);
   new (&problem->vars[9]) OP::Variable(2,11);
   new (&problem->vars[10]) OP::Variable(2,11);
   new (&problem->vars[11]) OP::Variable(1,1);

   problem->setStartEndLocations(0,1);

   problem->addPickupDelivery(2,3);
   problem->addPickupDelivery(4,5);
   problem->addPickupDelivery(6,7);
   problem->addPickupDelivery(8,9);
   problem->addPickupDelivery(10,11);

   unsigned int costs[12 * 12] = {
       0,    0,  449,  796,  112,  608,  885, 1016,  834, 1320,  640,  946,
       0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
       449,    0,    0,  518,  611,  795,  437,  568,  981, 1514,  192,  497,
       796,    0,  518,    0,  869,  880,  409,  371, 1066, 1315,  318,  515,
       112,    0,  611,  869,    0,  706,  960, 1091,  932, 1418,  715, 1042,
       608,    0,  795,  880,  706,    0, 1143,  938,  339,  796,  898, 1204,
       885,    0,  437,  409,  960, 1143,    0,  279, 1377, 1590,  327,  268,
       016,    0,  568,  371, 1091,  938,  279,    0, 1173, 1270,  359,  459,
       834,    0,  981, 1066,  932,  339, 1377, 1173,    0,  825, 1149, 1333,
       320,    0, 1514, 1315, 1418,  796, 1590, 1270,  825,    0, 1682, 1537,
       640,    0,  192,  318,  715,  898,  327,  359, 1149, 1682,    0,  382,
       946,    0,  497,  515, 1042, 1204,  268,  459, 1333, 1537,  382,    0
   };

   for(unsigned int i = 0; i < 12*12; i+= 1)
   {
     problem->distances[i] = costs[i];
   }
    */
}

__global__
void initRoots(MDD::MDD::Type type, unsigned int width, unsigned int* rootsCount, DP::TSPState* roots, unsigned int* levels, OP::TSPProblem const * problem, unsigned int cutsetSize, DP::TSPState * cutsetStates)
{
    __shared__ unsigned int alignedSharedMem[1000];
    std::byte* sharedMem = reinterpret_cast<std::byte*>(alignedSharedMem);

    ONE_THREAD_IN_BLOCK
    {
        //MDD
        auto start = Chrono::now();
        MDD::MDD mdd(type, width, roots, 0, problem, cutsetStates);
        mdd.buildTopDown(sharedMem);
        auto end = Chrono::now();
        printf("[INFO] Master MDD created in %d ms\n", static_cast<unsigned int>(end - start));

        mdd.print();

        //Copy states for next iteration
        start = Chrono::now();
        unsigned int rootIdx  = 0;
        for (unsigned int level = 0; level < mdd.dag.height; level += 1)
        {
            DP::TSPState* states = mdd.dag.getStates(level);
            for(unsigned int stateIdx = 0; stateIdx < mdd.dag.width; stateIdx += 1)
            {
                DP::TSPState const & state = states[stateIdx];
                if (state.active and state.exact)
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

        for(int i  = 0; i < cutsetSize; i += 1)
        {
            if (cutsetStates[i].active)
            {
                printf("[INFO] Cutset state with cost %d is ", cutsetStates[i].cost);
                cutsetStates[i].admissibleValues.print();
            }
            else
            {
                break;
            }

        }

    }
}

__global__
void buildMDDs(unsigned int width, DP::TSPState const * roots, unsigned int const * levels, unsigned int* bounds, OP::TSPProblem const * problem, unsigned int cutsetSize, DP::TSPState * cutsetsStates)
{
    __shared__ unsigned int alignedSharedMem[1000];
    std::byte* sharedMem = reinterpret_cast<std::byte*>(alignedSharedMem);

    ONE_THREAD_IN_BLOCK
    {
        //MDD
        auto start = Chrono::now();
        MDD::MDD mdd(MDD::MDD::Type::Relaxed, width, roots + blockIdx.x, levels[blockIdx.x], problem, cutsetsStates + (blockIdx.x * cutsetSize));
        auto end = Chrono::now();
        //printf("[INFO] MDD %d created in %d ms\n", blockIdx.x, static_cast<unsigned int>(end - start));


        start = Chrono::now();
        mdd.buildTopDown(sharedMem);
        bounds[2 * blockIdx.x] = mdd.getMinCost();
        mdd.type = MDD::MDD::Type::Restricted;
        mdd.dag.reset();
        mdd.buildTopDown(sharedMem);
        bounds[(2 * blockIdx.x) + 1] = mdd.getMinCost();
        end = Chrono::now();
        printf("[INFO] MDD %d built in %d ms [%d,%d]\n", blockIdx.x, static_cast<uint>(end - start), bounds[2 * blockIdx.x], bounds[(2 * blockIdx.x) + 1]);


        /*
        startLocation = Chrono::now();
        StaticVector<int> labels(mdd->height);
        mdd->DFS(0, 0,labels, true);
        end = Chrono::now();
        printf("[INFO] DFS performed in %d ms\n", static_cast<uint>(end - start));
        */
    }
}

