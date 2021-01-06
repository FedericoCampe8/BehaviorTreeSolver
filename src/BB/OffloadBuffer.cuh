#pragma once

#include <Containers/Buffer.cuh>
#include <Containers/MaxHeap.cuh>

#include "../DD/MDD.cuh"
#include "StateMetadata.cuh"

namespace BB
{
    template<typename StateType>
    class OffloadBuffer
    {
        // Members
        public:
            unsigned int const cutsetMaxSize;
        private:
            unsigned int size;
            Array<StateType> statesBuffer;
            Array<StateMetadata<StateType>> statesMetadataBuffer;
            Array<unsigned int> cutsetsSizes;
            Array<StateType> cutsetsBuffer;
            Array<StateType> approximateSolutionsBuffer;
            Array<LNS::Neighbourhood> neighbourhoods;

        // Functions
        public:
            template<typename ModelType, typename ProblemType>
            OffloadBuffer(DD::MDD<ModelType, ProblemType, StateType> const * mdd, unsigned int capacity, Memory::MallocType mallocType);
            void clear();
            void enqueue(StateMetadata<StateType> const * stateMetadata);
            void generateNeighbourhoods(StateType const * currentSolution, unsigned int eqPercentage, unsigned int neqPercentage, std::mt19937* rng);
            __host__ __device__ StateType * getApproximateSolution(unsigned int index) const;
            __host__ __device__ LightVector<StateType> getCutset(unsigned int index) const;
            __host__ __device__ LNS::Neighbourhood const * getNeighbourhood(unsigned int index) const;
            unsigned int getSize() const;
            __host__ __device__ StateMetadata<StateType> * getStateMetadata(unsigned int index) const;
            bool isEmpty() const;
            bool isFull() const;
    };

    template<typename StateType>
    template<typename ModelType, typename ProblemType>
    OffloadBuffer<StateType>::OffloadBuffer(DD::MDD<ModelType, ProblemType, StateType> const * mdd, unsigned int capacity, Memory::MallocType mallocType) :
        cutsetMaxSize(mdd->calcCutsetMaxSize()),
        statesBuffer(capacity, mallocType),
        statesMetadataBuffer(capacity, mallocType),
        cutsetsSizes(capacity, mallocType),
        cutsetsBuffer(cutsetMaxSize * capacity, mallocType),
        approximateSolutionsBuffer(capacity, mallocType),
        neighbourhoods(capacity, mallocType)
    {
        ProblemType const * const problem = mdd->model->problem;
        unsigned int storageSize = StateType::sizeOfStorage(problem);

        // States
        std::byte* storages = StateType::mallocStorages(problem, statesBuffer.getCapacity(), mallocType);
        for (unsigned int stateIdx = 0; stateIdx < statesBuffer.getCapacity(); stateIdx += 1)
        {
            new (statesBuffer[stateIdx]) StateType(problem, &storages[storageSize * stateIdx]);
        }

        // Cutsets
        storages = StateType::mallocStorages(problem, cutsetsBuffer.getCapacity(), mallocType);
        for (unsigned int stateIdx = 0; stateIdx < cutsetsBuffer.getCapacity(); stateIdx += 1)
        {
            new (cutsetsBuffer[stateIdx]) StateType(problem, &storages[storageSize * stateIdx]);
        }

        // Approximate solutions
        storages = StateType::mallocStorages(problem, approximateSolutionsBuffer.getCapacity(), mallocType);
        for (unsigned int stateIdx = 0; stateIdx < approximateSolutionsBuffer.getCapacity(); stateIdx += 1)
        {
            new (approximateSolutionsBuffer[stateIdx]) StateType(problem, &storages[storageSize * stateIdx]);
        }

        // Neighbourhood
        storageSize = LNS::Neighbourhood::sizeOfStorage(problem);
        storages = LNS::Neighbourhood::mallocStorages(problem, neighbourhoods.getCapacity(), mallocType);
        for (unsigned int neighbourhoodIdx = 0; neighbourhoodIdx < neighbourhoods.getCapacity(); neighbourhoodIdx += 1)
        {
            new (neighbourhoods[neighbourhoodIdx]) LNS::Neighbourhood(problem, &storages[storageSize * neighbourhoodIdx]);
        }
    }

    template<typename StateType>
    void OffloadBuffer<StateType>::clear()
    {
        size = 0;
    }

    template<typename StateType>
    void OffloadBuffer<StateType>::enqueue(StateMetadata<StateType> const * stateMetadata)
    {
        unsigned int const index = size - 1;
        *statesBuffer[index] = *stateMetadata->state;
        new (statesMetadataBuffer[index]) StateMetadata<StateType>(stateMetadata->lowerbound, stateMetadata->upperbound, statesBuffer[index]);
        *cutsetsSizes[index] = 0;
        size += 1;
    }

    template<typename StateType>
    void OffloadBuffer<StateType>::generateNeighbourhoods(StateType const * currentSolution, unsigned int eqPercentage, unsigned int neqPercentage, std::mt19937* rng)
    {
        for (unsigned int index = 0; index < size; index += 1)
        {
            neighbourhoods[index]->generate(&currentSolution->selectedValues, eqPercentage, neqPercentage, rng);
        }
    }

    template<typename StateType>
    __host__ __device__
    StateType * OffloadBuffer<StateType>::getApproximateSolution(unsigned int index) const
    {
        return approximateSolutionsBuffer[index];
    }

    template<typename StateType>
    __host__ __device__
    LightVector<StateType> OffloadBuffer<StateType>::getCutset(unsigned int index) const
    {
        return LightVector<StateType>(*cutsetsSizes[index], cutsetsBuffer[cutsetMaxSize * index]);
    }

    template<typename StateType>
    __host__ __device__
    LNS::Neighbourhood const* OffloadBuffer<StateType>::getNeighbourhood(unsigned int index) const
    {
        return neighbourhoods[index];
    }

    template<typename StateType>
    unsigned int OffloadBuffer<StateType>::getSize() const
    {
        return size;
    }

    template<typename StateType>
    __host__ __device__
    StateMetadata<StateType> * OffloadBuffer<StateType>::getStateMetadata(unsigned int index) const
    {
        return statesMetadataBuffer[index];
    }

    template<typename StateType>
    bool OffloadBuffer<StateType>::isEmpty() const
    {
        return size == 0;
    }

    template<typename StateType>
    bool OffloadBuffer<StateType>::isFull() const
    {
        return size == statesMetadataBuffer.getCapacity();
    }
}





