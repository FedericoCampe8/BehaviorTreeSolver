#pragma once
#include <thread>
#include <random>
#include <curand_kernel.h>
#include <LNS/OffloadBuffer.cuh>
#include <Options.h>
#include <Utils/CUDA.cuh>
#include <Utils/CUDA.cuh>
#include <thrust/extrema.h>

template<typename ProblemType, typename StateType>
class OffloadBufferGPU : public OffloadBuffer<ProblemType,StateType>
{
    // Members
    private:
    Array<curandStatePhilox4_32_10_t> rngs;

    // Functions
    public:
    OffloadBufferGPU(ProblemType const * problem, Options const & options, Memory::MallocType mallocType);
    void initializeRngsAsync(u32 randomSeed);
    void generateNeighbourhoodsAsync(StateType const * solution);
    void wait();
    void doOffloadAsync(LNS::SearchPhase searchPhase);
};

__global__
void initializeRngKernel(Array<curandStatePhilox4_32_10_t>* rngs, u32 randomSeed)
{
    if(threadIdx.x == 0)
    {
        u32 const rngIdx = blockIdx.x;
        curand_init(randomSeed + rngIdx, 0, 0, rngs->at(rngIdx));
    }
}

template<typename StateType>
__global__
void generateNeighbourhoodKernel(Array<curandStatePhilox4_32_10_t>* rngs, Array<Neighbourhood>* neighbourhoods, StateType const * solution)
{
    if(threadIdx.x == 0)
    {
        u32 const index = blockIdx.x;
        neighbourhoods->at(index)->reset();
        for(u32 variableIdx = 0; variableIdx < solution->selectedValues.getCapacity(); variableIdx += 1)
        {
            OP::ValueType const value = *solution->selectedValues[variableIdx];
            float const random = curand_uniform(rngs->at(index));
            neighbourhoods->at(index)->constraintVariable(variableIdx, value, random);
        }
    }
}

template<typename ProblemType, typename StateType>
__global__
void doOffloadKernel(Array<DD::MDD<ProblemType, StateType>>* mdds, Array<Neighbourhood>* neighbourhoods, Array<StateType>* topStates, Array<StateType>* bottomStates, LNS::SearchPhase searchPhase)
{
    u32 const index= blockIdx.x;
    if(searchPhase == LNS::SearchPhase::Init)
    {
        mdds->at(index)->buildTopDown(neighbourhoods->at(index), topStates->at(index), bottomStates->at(index), false);
    }
    else
    {
        mdds->at(index)->buildTopDown(neighbourhoods->at(index), topStates->at(0), bottomStates->at(index), true);
    }
}

template<typename ProblemType, typename StateType>
OffloadBufferGPU<ProblemType, StateType>::OffloadBufferGPU(ProblemType const * problem, Options const & options, Memory::MallocType mallocType) :
    OffloadBuffer<ProblemType, StateType>(problem, options.widthGpu, options.parallelismGpu, options.eqProbability, options.neqProbability, mallocType),
    rngs(this->capacity, mallocType)
{}

template<typename ProblemType, typename StateType>
void OffloadBufferGPU<ProblemType, StateType>::initializeRngsAsync(u32 randomSeed)
{
    u32 const blockSize = 1;
    u32 const blockCount = this->capacity;
    initializeRngKernel<<<blockCount, blockSize>>>(&rngs, randomSeed);
}

template<typename ProblemType, typename StateType>
void OffloadBufferGPU<ProblemType, StateType>::generateNeighbourhoodsAsync(StateType const * solution)
{
    if(not this->isEmpty())
    {
        u32 const blockSize = 1;
        u32 const blockCount = this->capacity;
        generateNeighbourhoodKernel<<<blockCount, blockSize>>>(&rngs, &this->neighbourhoods, solution);
    }
}

template<typename ProblemType, typename StateType>
void OffloadBufferGPU<ProblemType, StateType>::wait()
{
    if(not this->isEmpty())
    {
        cudaDeviceSynchronize();
    }
}

template<typename ProblemType, typename StateType>
void OffloadBufferGPU<ProblemType, StateType>::doOffloadAsync(LNS::SearchPhase searchPhase)
{
    if(not this->isEmpty())
    {
        u32 const blockSize = Algorithms::nextPower2(this->mdds[0]->width * this->mdds[0]->problem->maxBranchingFactor);
        u32 const blockCount =  searchPhase == LNS::SearchPhase::Init ? this->size : this->capacity;
        u32 const sharedMemSize = this->mdds[0]->sizeOfScratchpadMemory();
        assert(blockSize <= 1024);
        doOffloadKernel<<<blockCount, blockSize, sharedMemSize>>>(&this->mdds, &this->neighbourhoods, &this->topStates, &this->bottomStates, searchPhase);
    }
}