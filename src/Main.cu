#include <cstdio>
#include <cstdlib>
#include <new>

#include <MDD/MDD.cuh>
#include <Problem/AllDifferent.cuh>
#include <Problem/Variable.cuh>
#include <Extra/Utils/Chrono.cuh>
#include <Extra/Utils/CUDA.cuh>
#include <Extra/Utils/Memory.cuh>

//Auxiliary functions
Variable * mallocVars(unsigned int varsCount);
void initVars(unsigned int varsCount, Variable * vars);
__global__ void buildMDD(unsigned int width, unsigned int varsCount, Variable * vars);

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

    // Variables
    unsigned int varsCount = std::atoi(argv[1]);
    Variable * vars = mallocVars(varsCount);
    initVars(varsCount,vars);

    //Build MDDs
    unsigned int width = std::atoi(argv[2]);
    unsigned int copies = std::atoi(argv[3]);
    auto start = Chrono::now();
    buildMDD<<<copies,1>>>(width, varsCount, vars);
    cudaDeviceSynchronize();
    auto end = Chrono::now();
    printf("[INFO] Kernel executed in %d ms\n",  static_cast<uint>(end - start));
    return EXIT_SUCCESS;
}

Variable * mallocVars(unsigned int varsCount)
{
    Variable * vars;
    cudaMallocManaged(&vars, sizeof(Variable) * varsCount);
    return vars;
}


void initVars(unsigned int varsCount, Variable * vars)
{
    for(unsigned int i = 0; i < varsCount; i +=1)
    {
        new (&vars[i]) Variable(0, varsCount - 1);
    }
}

__global__
void buildMDD(unsigned int width, unsigned int varsCount, Variable * vars)
{
    using State = AllDifferent::State;
    __shared__ MDD<State> * mdd;
    __shared__ uint32_t shrMem[1000];

    if(threadIdx.x == 0)
    {
        auto start = Chrono::now();
        mdd = static_cast<MDD<State>*>(Memory::alignedMalloc(sizeof(MDD<State>)));
        new (mdd) MDD<State>(MDD<State>::Type::Restricted, width, varsCount, vars);
        auto end = Chrono::now();
        //printf("[INFO] MDD %d created in %d ms\n", blockIdx.x, static_cast<unsigned int>(end - start));


        start = Chrono::now();
        mdd->buildTopDown(reinterpret_cast<std::byte*>(shrMem));
        end = Chrono::now();
        //printf("[INFO] MDD %d built in %d ms\n", blockIdx.x, static_cast<uint>(end - start));

        /*
        start = Chrono::now();
        StaticVector<int> labels(mdd->height);
        mdd->DFS(0, 0,labels, true);
        end = Chrono::now();
        printf("[INFO] DFS performed in %d ms\n", static_cast<uint>(end - start));
        */

        //mdd->toGraphViz();
    }
}
