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
            Vector<Array<unsigned int>*> bufferToQueues;
            Vector<MaxHeap<QueuedState<StateType>>*> queues;

        // Functions
        public:
            template<typename ModelType, typename ProblemType>
            PriorityQueuesManager(ProblemType const * problem, unsigned int queuesSize, unsigned int queuesCount);
            bool areQueuesEmpty();
            void dequeue(QueuedState<StateType> const * queuedState);
            void enqueue(unsigned int lowerbound, unsigned int upperbound, StateType const * state);
            unsigned int getQueuesSize();
            void registerQueue(MaxHeap<QueuedState<StateType>>* queue);
    };

    template<typename StateType>
    template<typename ProblemType>
    PriorityQueuesManager<StateType>::PriorityQueuesManager(ProblemType const * problem, unsigned int queuesSize, unsigned int queuesCount) :
        Buffer(queuesSize, Memory::MallocType::Std),
        bufferToQueues(queuesCount, Memory::MallocType::Std),
        queues(queuesCount, Memory::MallocType::Std)
    {
        unsigned int const storageSize =  StateType::sizeOfStorage(problem);
        std::byte* const storages = StateType::mallocStorages(getCapacity(), problem, Memory::MallocType::Std);
        for(unsigned int stateIdx = 0; stateIdx < getCapacity(); stateIdx += 1)
        {
            new (at(stateIdx)) StateType(problem, &storages[storageSize * stateIdx]);
        }
    }

    template<typename StateType>
    bool PriorityQueuesManager<StateType>::areQueuesEmpty()
    {
       return isEmpty();
    }

    template<typename StateType>
    void PriorityQueuesManager<StateType>::dequeue(QueuedState<StateType> const * queuedState)
    {
        unsigned int const bufferIdx = indexOf(queuedState->state);
        for(unsigned int queueIdx = 0; queueIdx < queues.getSize(); queueIdx += 1)
        {
            MaxHeap<QueuedState<StateType>>* queue = queues[queueIdx];
            unsigned int const index = *((*bufferToQueues[queueIdx])->at(bufferIdx));
            queue->erase(index);
            queue->popBack();
        }
    }

    template<typename StateType>
    void PriorityQueuesManager<StateType>::enqueue(unsigned int lowerbound, unsigned int upperbound, StateType const * state)
    {
        StateType const * const bufferedState = insert(state);
        unsigned int const bufferIdx = indexOf(bufferedState);

        for(unsigned int queueIdx = 0; queueIdx < queues.getSize(); queueIdx += 1)
        {
            MaxHeap<QueuedState<StateType>>* queue = queues[queueIdx];
            unsigned int * const index = (*bufferToQueues[queueIdx])->at(bufferIdx);
            *index = queue->getSize();
            queue->emplaceBack(index, lowerbound, upperbound, bufferedState);
            queue->insert();
        }
    }

    template<typename StateType>
    unsigned int PriorityQueuesManager<StateType>::getQueuesSize()
    {
        return getSize();
    }

    template<typename StateType>
    void PriorityQueuesManager<StateType>::registerQueue(MaxHeap<QueuedState<StateType>>* queue)
    {
        assert(getCapacity() == queue->getCapacity());

        bufferToQueues.emplaceBack(queue->getCapacity(), Memory::MallocType::Std);
        queues.pushBack(&queue);
    }
}