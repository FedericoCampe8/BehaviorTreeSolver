#pragma once

#include <thrust/distance.h>
#include <Containers/Buffer.cuh>
#include <Containers/MaxHeap.cuh>

#include "../DD/MDD.cuh"
#include "OffloadedState.cuh"

namespace BB
{
    template<typename T>
    class OffloadQueue
    {
        private:
            using StateType = T::StateType;
            using OffloadedStateType = BB::OffloadedState<StateType>;

        public:
            unsigned int cutsetMaxSize;
            Vector<OffloadedStateType> queue;
        private:
            Array<StateType> statesBuffer;
            Array<StateType> cutsetsBuffer;
            Array<StateType> upperboundStatesBuffer;

        public:
            OffloadQueue(DD::MDD<T> const & mdd, unsigned int offloadMaxSize, Memory::MallocType mallocType);
            void enqueue(StateType const & state);
    };



    template<typename T>
    OffloadQueue<T>::OffloadQueue(DD::MDD<T> const & mdd, unsigned int offloadMaxSize, Memory::MallocType mallocType) :
        cutsetMaxSize(mdd.width * mdd.fanout),
        queue(offloadMaxSize, mallocType),
        statesBuffer(offloadMaxSize, mallocType),
        cutsetsBuffer(cutsetMaxSize * offloadMaxSize, mallocType),
        upperboundStatesBuffer(offloadMaxSize, mallocType),
    {
        T::ProblemType const& problem = mdd.model.problem;
        unsigned int storageSize = StateType::sizeOfStorage(problem);

        // States
        std::byte* storages = StateType::mallocStorages(statesBuffer.getCapacity(), problem, mallocType);
        for (unsigned int stateIdx = 0; stateIdx < statesBuffer.getCapacity(); stateIdx += 1)
        {
            new (&statesBuffer[stateIdx]) StateType(problem, &storages[storageSize * stateIdx]);
        }

        // Cutsets
        storages = StateType::mallocStorages(cutsetsBuffer.getCapacity(), problem, mallocType);
        for (unsigned int stateIdx = 0; stateIdx < cutsetsBuffer.getCapacity(); stateIdx += 1)
        {
            new (&cutsetsBuffer[stateIdx]) StateType(problem, &storages[storageSize * stateIdx]);
        }

        // Upperbounds
        storages = StateType::mallocStorages(upperboundStatesBuffer.getCapacity(), problem, mallocType);
        for (unsigned int stateIdx = 0; stateIdx < upperboundStatesBuffer.getCapacity(); stateIdx += 1)
        {
            new(&upperboundStatesBuffer[stateIdx]) StateType(problem, &storages[storageSize * stateIdx]);
        }
    }

    template<typename T>
    void OffloadQueue<T>::doOffloadCpu()
    {
        thrust::for_each(thrust::host, queue.begin(), queue.end(), [&] (OffloadedStateType& offloadedState)
        {
            unsigned int stateIdx = thrust::distance(queue.begin(), &offloadedState);
            doOffload(mdd, offloadedState, &scratchpadMem[mdd.scratchpadMemSize * stateIdx]);
        });
    }

    template<typename T>
    void OffloadQueue<T>::doOffloadGpU()
    {
        doOffloadKernel<<<queue.getSize(),1,mdd.scratchpadMemSize>>>(mdd, queue);
        cudaDeviceSynchronize();
    }


    template<typename T>
    void OffloadQueue<T>::enqueue(StateType const & state)
    {
        unsigned int const index = queue.getSize();
        statesBuffer[index] = state;

        queue.resize(queue.getSize() + 1);
        new (&queue.back()) OffloadedStateType(cutsetMaxSize, &cutsetsBuffer[cutsetMaxSize * index], upperboundStatesBuffer[index], statesBuffer[index]);
    }
}





