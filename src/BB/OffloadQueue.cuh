#pragma once

#include <thrust/distance.h>
#include <Containers/Buffer.cuh>
#include <Containers/MaxHeap.cuh>

#include "../DD/MDD.cuh"
#include "OffloadedState.cuh"

namespace BB
{
    template<typename StateType>
    class OffloadQueue : public Vector<OffloadedState<StateType>>
    {
        // Members
        public:
            unsigned int const cutsetMaxSize;
        private:
            Array<StateType> statesBuffer;
            Array<StateType> cutsetsBuffer;
            Array<StateType> upperboundStatesBuffer;

        // Functions
        public:
            template<typename ModelType, typename ProblemType>
            OffloadQueue(DD::MDD<ModelType,ProblemType,StateType> const * mdd, unsigned int offloadMaxSize, Memory::MallocType mallocType);
            void enqueue(StateType const * state);
    };

    template<typename StateType>
    template<typename ModelType, typename ProblemType>
    OffloadQueue<StateType>::OffloadQueue(DD::MDD<ModelType,ProblemType,StateType> const * mdd, unsigned int offloadMaxSize, Memory::MallocType mallocType) :
        Vector<OffloadedState<StateType>>(offloadMaxSize, mallocType),
        cutsetMaxSize(mdd->cutsetMaxSize),
        statesBuffer(offloadMaxSize, mallocType),
        cutsetsBuffer(cutsetMaxSize * offloadMaxSize, mallocType),
        upperboundStatesBuffer(offloadMaxSize, mallocType)
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

        // Upperbounds
        storages = StateType::mallocStorages(problem, upperboundStatesBuffer.getCapacity(), mallocType);
        for (unsigned int stateIdx = 0; stateIdx < upperboundStatesBuffer.getCapacity(); stateIdx += 1)
        {
            new (upperboundStatesBuffer[stateIdx]) StateType(problem, &storages[storageSize * stateIdx]);
        }
    }

    template<typename StateType>
    void OffloadQueue<StateType>::enqueue(StateType const * state)
    {
        unsigned int const index = this->size;
        *statesBuffer[index] = *state;

        this->incrementSize();
        new (this->back()) OffloadedState<StateType>(statesBuffer[index], cutsetMaxSize, cutsetsBuffer[cutsetMaxSize * index], upperboundStatesBuffer[index]);
    }
}





