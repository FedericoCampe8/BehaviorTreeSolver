#pragma once

#include <thrust/distance.h>

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
            Vector<Array<uint32_t>> bufferToQueues;
            Vector<MaxHeap<QueuedState<StateType>>*> queues;

        // Functions
        public:
            template<typename ProblemType>
            PriorityQueuesManager(ProblemType const * problem, unsigned int queuesSize, unsigned int queuesCount);
            bool areQueuesEmpty();
            void dequeue(QueuedState<StateType> const * queuedState);
            void enqueue(unsigned int lowerbound, unsigned int upperbound, StateType const * state);
            unsigned int getQueuesSize();
            void registerQueue(MaxHeap<QueuedState<StateType>>* queue);
        private:
            bool checkQueues();
            static void printQueue(MaxHeap<QueuedState<StateType>>* queue);
    };

    template<typename StateType>
    template<typename ProblemType>
    PriorityQueuesManager<StateType>::PriorityQueuesManager(ProblemType const * problem, unsigned int queuesSize, unsigned int queuesCount) :
        Buffer<StateType>(queuesSize, Memory::MallocType::Std),
        bufferToQueues(queuesCount, Memory::MallocType::Std),
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
    bool PriorityQueuesManager<StateType>::areQueuesEmpty()
    {
       return this->isEmpty();
    }

    template<typename StateType>
    void PriorityQueuesManager<StateType>::dequeue(QueuedState<StateType> const * queuedState)
    {
        unsigned int const bufferIdx = this->indexOf(queuedState->state);

        for(unsigned int queueIdx = 0; queueIdx < queues.getSize(); queueIdx += 1)
        {
            MaxHeap<QueuedState<StateType>>* queue = *queues[queueIdx];
            unsigned int const index = *bufferToQueues[queueIdx]->at(bufferIdx);
            queue->erase(index);
        }
        this->erase(queuedState->state);
    }

    template<typename StateType>
    void PriorityQueuesManager<StateType>::enqueue(unsigned int lowerbound, unsigned int upperbound, StateType const * state)
    {
        StateType const * const bufferedState = this->insert(state);
        unsigned int const bufferIdx = this->indexOf(bufferedState);

        assert(checkQueues());

        for (unsigned int queueIdx = 0; queueIdx < queues.getSize(); queueIdx += 1)
        {
            MaxHeap<QueuedState<StateType>>* queue = *queues[queueIdx];
            unsigned int * const heapIdx = bufferToQueues[queueIdx]->at(bufferIdx);
            *heapIdx = queue->getSize();
            queue->incrementSize();
            new (queue->back()) QueuedState<StateType>(heapIdx, lowerbound, upperbound, bufferedState);
            queue->insertBack();
        }

        assert(checkQueues());
    }

    template<typename StateType>
    unsigned int PriorityQueuesManager<StateType>::getQueuesSize()
    {
        return this->getSize();
    }

    template<typename StateType>
    void PriorityQueuesManager<StateType>::registerQueue(MaxHeap<QueuedState<StateType>>* queue)
    {
        assert(this->getCapacity() == queue->getCapacity());
        bufferToQueues.incrementSize();
        new (bufferToQueues.back()) Array<unsigned int>(queue->getCapacity(), Memory::MallocType::Std);
        queues.incrementSize();
        *queues.back() = queue;
    }

    template<typename StateType>
    bool PriorityQueuesManager<StateType>::checkQueues()
    {
        for (unsigned int queueIdx = 0; queueIdx < queues.getSize(); queueIdx += 1)
        {
            MaxHeap<QueuedState<StateType>>* queue = *queues[queueIdx];
            for (unsigned int queuedStateIdx = 0; queuedStateIdx < queue->getSize(); queuedStateIdx += 1)
            {
                assert(*queue->at(queuedStateIdx)->heapIdx == queuedStateIdx);
            }
        }

        return true;
    }
    template<typename StateType>
    void PriorityQueuesManager<StateType>::printQueue(MaxHeap<QueuedState<StateType>>* queue)
    {
        if(not queue->isEmpty())
        {
            printf("[DEBUG]----\n");
            for (unsigned int queuedStateIdx = 0; queuedStateIdx < queue->getSize(); queuedStateIdx += 1)
            {
                QueuedState<StateType>* queuedState = queue->at(queuedStateIdx);
                printf("[DEBUG] State: ");
                queuedState->state->selectedValues.print(false);
                printf(" | Index: %d | Self Idx: %u\n", queuedStateIdx, *queuedState->heapIdx);
            }
        }
        else
        {
            printf("[DEBUG] Empty queue\n");
        }
    }
}