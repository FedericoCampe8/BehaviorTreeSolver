#pragma once

#include <bitset>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <type_traits>
#include <thrust/distance.h>
#include <thrust/equal.h>
#include <Containers/Buffer.cuh>
#include <Containers/MaxHeap.cuh>

#include "../DD/MDD.cuh"
#include "QueuedState.cuh"

namespace BB
{
    template<typename StateType>
    class PriorityQueuesManager : public Buffer<StateType>
    {
        // Members
        private:
            Vector<Array<QueuedState<StateType>*>> bufferIdxToQueue;
            Vector<MaxHeap<QueuedState<StateType>>*> queues;

        // Functions
        public:
            template<typename ProblemType>
            PriorityQueuesManager(ProblemType const * problem, unsigned int queuesSize, unsigned int queuesCount);
            void dequeue(StateType const * state);
            void enqueue(StateMetadata<StateType> const * stateMetadata);
            void registerQueue(MaxHeap<QueuedState<StateType>>* queue);
    };

    template<typename StateType>
    template<typename ProblemType>
    PriorityQueuesManager<StateType>::PriorityQueuesManager(ProblemType const * problem, unsigned int queuesSize, unsigned int queuesCount) :
        Buffer<StateType>(queuesSize, Memory::MallocType::Std),
        bufferIdxToQueue(queuesCount, Memory::MallocType::Std),
        queues(queuesCount, Memory::MallocType::Std)
    {
        unsigned int const storageSize =  StateType::sizeOfStorage(problem);
        std::byte* const storages = StateType::mallocStorages(problem, this->capacity, Memory::MallocType::Std);
        for(unsigned int stateIdx = 0; stateIdx < this->capacity; stateIdx += 1)
        {
            new (this->at(stateIdx)) StateType(problem, &storages[storageSize * stateIdx]);
        }
    }

    template<typename StateType>
    void PriorityQueuesManager<StateType>::dequeue(StateType const * state)
    {
        unsigned int const bufferIdx = this->indexOf(state);

        for(unsigned int queueIdx = 0; queueIdx < queues.getSize(); queueIdx += 1)
        {
            MaxHeap<QueuedState<StateType>>* queue = *queues[queueIdx];
            queue->erase(*bufferIdxToQueue[queueIdx]->at(bufferIdx));
        }
        this->erase(state);
    }

    template<typename StateType>
    void PriorityQueuesManager<StateType>::enqueue(StateMetadata<StateType> const * stateMetadata)
    {
        StateType const * const bufferState = this->insert(stateMetadata->state);
        StateMetadata<StateType> const bufferStateMetadata(stateMetadata->lowerbound, stateMetadata->upperbound, bufferState);
        unsigned int const bufferIdx = this->indexOf(bufferState);
        for (unsigned int queueIdx = 0; queueIdx < queues.getSize(); queueIdx += 1)
        {
            MaxHeap<QueuedState<StateType>>* queue = *queues[queueIdx];
            queue->incrementSize();
            *bufferIdxToQueue[queueIdx]->at(bufferIdx) = queue->back();
            new (queue->back()) QueuedState<StateType>(&bufferStateMetadata, bufferIdxToQueue[queueIdx]->at(bufferIdx));
            queue->insertBack();
        }
    }

    template<typename StateType>
    void PriorityQueuesManager<StateType>::registerQueue(MaxHeap<QueuedState<StateType>>* queue)
    {
        assert(this->getCapacity() == queue->getCapacity());
        bufferIdxToQueue.incrementSize();
        new (bufferIdxToQueue.back()) Array<QueuedState<StateType>*>(queue->getCapacity(), Memory::MallocType::Std);
        queues.incrementSize();
        *queues.back() = queue;
    }
}