#pragma once

#include <thrust/distance.h>

#include <Containers/Buffer.cuh>
#include <Containers/MaxHeap.cuh>

#include "../DD/MDD.cuh"
#include "QueuedState.cuh"

namespace BB
{
    template<typename StateType>
    class QueuesManager
    {
        public:
            typedef bool (*Comparator) (QueuedState<StateType> const & s0, QueuedState<StateType> const & s1);

        private:
            Buffer<StateType> statesBuffer;
            Vector<Array<unsigned int>> bufferToQueues;
            Vector<MaxHeap<QueuedState<StateType>>*> queues;

        public:
            template<typename ModelType, typename ProblemType>
            QueuesManager(DD::MDD<ModelType, ProblemType, StateType> const & mdd, unsigned int queuesSize, unsigned int queuesCount);
            bool areQueuesEmpty();
            void dequeue(QueuedState<StateType> const & queuedstate);
            void enqueue(unsigned int lowerbound, unsigned int upperbound, StateType const & state);
            unsigned int getQueuesSize();
            void registerQueue(MaxHeap<QueuedState<StateType>>& queue);
    };

    template<typename StateType>
    template<typename ModelType, typename ProblemType>
    QueuesManager<StateType>::QueuesManager(DD::MDD<ModelType, ProblemType, StateType> const & mdd, unsigned int queuesSize, unsigned int queuesCount) :
        statesBuffer(queuesSize, Memory::MallocType::Std),
        bufferToQueues(queuesCount, Memory::MallocType::Std),
        queues(queuesCount, Memory::MallocType::Std)
    {
        ProblemType const & problem = mdd.model.problem;
        unsigned int storageSize =  StateType::sizeOfStorage(problem);
        std::byte* storages = StateType::mallocStorages(statesBuffer.getCapacity(), problem, Memory::MallocType::Std);
        for(unsigned int stateIdx = 0; stateIdx < statesBuffer.getCapacity(); stateIdx += 1)
        {
            new (&statesBuffer[stateIdx]) StateType(problem, &storages[storageSize * stateIdx]);
        }
    }

    template<typename StateType>
    bool QueuesManager<StateType>::areQueuesEmpty()
    {
       return statesBuffer.isEmpty();
    }

    template<typename StateType>
    void QueuesManager<StateType>::dequeue(QueuedState<StateType> const & queuesState)
    {
        unsigned int const bufferIdx = statesBuffer.indexOf(queuesState.state);
        for(unsigned int queueIdx = 0; queueIdx < queues.getSize(); queueIdx += 1)
        {
            MaxHeap<QueuedState<StateType>>& queue = *queues[queueIdx];
            unsigned int const index = bufferToQueues[queueIdx][bufferIdx];
            queue.erase(index);
            queue.popBack();
        }
    }

    template<typename StateType>
    void QueuesManager<StateType>::enqueue(unsigned int lowerbound, unsigned int upperbound, StateType const & state)
    {
        StateType& bufferedState = statesBuffer.insert(state);
        unsigned int const bufferIdx = statesBuffer.indexOf(bufferedState);

        for(unsigned int queueIdx = 0; queueIdx < queues.getSize(); queueIdx += 1)
        {
            MaxHeap<QueuedState<StateType>>& queue = *queues[queueIdx];
            unsigned int * const index = &bufferToQueues[queueIdx][bufferIdx];
            *index = queue.getSize();
            queue.emplaceBack(index, lowerbound, upperbound, bufferedState);
            queue.insert();
        }
    }

    template<typename StateType>
    unsigned int QueuesManager<StateType>::getQueuesSize()
    {
        return statesBuffer.getSize();
    }

    template<typename StateType>
    void QueuesManager<StateType>::registerQueue(MaxHeap<QueuedState<StateType>>& queue)
    {
        assert(statesBuffer.getCapacity() == queue.getCapacity());

        bufferToQueues.emplaceBack(queue.getCapacity(), Memory::MallocType::Std);
        queues.pushBack(&queue);
    }
}