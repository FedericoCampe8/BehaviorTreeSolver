#pragma once

#include <Containers/Buffer.cuh>
#include <Containers/MaxHeap.cuh>

#include "../DD/MDD.cuh"
#include "StateMetadata.cuh"

namespace BB
{
    template<typename StateType>
    class PriorityQueue
    {
        // Members
        private:
            Buffer<StateType> statesBuffer;
            MaxHeap<BB::StateMetadata<StateType>> queue;

        // Functions
        public:
            template<typename ProblemType>
            PriorityQueue(ProblemType const * problem, typename MaxHeap<BB::StateMetadata<StateType>>::Comparator comparator, unsigned int capacity);
            void erase(BB::StateMetadata<StateType> const * stateMetadata);
            BB::StateMetadata<StateType> const * front() const;
            void insert(BB::StateMetadata<StateType> const * stateMetadata);
            bool isEmpty() const;
            bool isFull() const;
    };

    template<typename StateType>
    template<typename ProblemType>
    PriorityQueue<StateType>::PriorityQueue(ProblemType const * problem, typename MaxHeap<BB::StateMetadata<StateType>>::Comparator comparator, unsigned int capacity) :
        statesBuffer(capacity, Memory::MallocType::Std),
        queue(comparator, capacity, Memory::MallocType::Std)
    {
        unsigned int const storageSize =  StateType::sizeOfStorage(problem);
        std::byte* const memory = StateType::mallocStorages(problem, capacity, Memory::MallocType::Std);
        for(unsigned int stateIdx = 0; stateIdx < statesBuffer.getCapacity(); stateIdx += 1)
        {
            new (statesBuffer[stateIdx]) StateType(problem, &memory[storageSize * stateIdx]);
        }
    }

    template<typename StateType>
    void PriorityQueue<StateType>::erase(const StateMetadata<StateType>* stateMetadata)
    {
        queue.erase(stateMetadata);
        statesBuffer.erase(stateMetadata->state);
    }

    template<typename StateType>
    StateMetadata<StateType> const * PriorityQueue<StateType>::front() const
    {
        return queue.front();
    }

    template<typename StateType>
    void PriorityQueue<StateType>::insert(BB::StateMetadata<StateType> const * stateMetadata)
    {
        StateType const * const bufferedState = statesBuffer.insert(stateMetadata->state);
        StateMetadata<StateType> const bufferedStateMetadata(stateMetadata->lowerbound, stateMetadata->upperbound, bufferedState);
        queue.pushBack(&bufferedStateMetadata);
        queue.insertBack();
    }

    template<typename StateType>
    bool PriorityQueue<StateType>::isEmpty() const
    {
        return queue.isEmpty();
    }

    template<typename StateType>
    bool PriorityQueue<StateType>::isFull() const
    {
        return queue.isFull();
    }
}