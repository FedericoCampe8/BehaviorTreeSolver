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
            Array<std::unordered_map<std::size_t, unsigned int>> bestsCosts;

        // Functions
        public:
            template<typename ProblemType>
            PriorityQueuesManager(ProblemType const * problem, unsigned int queuesSize, unsigned int queuesCount);
            bool areQueuesEmpty();
            void dequeue(StateType const * state);
            void enqueue(StateMetadata<StateType> const * stateMetadata);
            unsigned int getQueuesSize();
            void reduceQueuesByClass();
            void registerQueue(MaxHeap<QueuedState<StateType>>* queue);

        private:
            static std::size_t hash(StateType const * state);
    };

    template<typename StateType>
    template<typename ProblemType>
    PriorityQueuesManager<StateType>::PriorityQueuesManager(ProblemType const * problem, unsigned int queuesSize, unsigned int queuesCount) :
        Buffer<StateType>(queuesSize, Memory::MallocType::Std),
        bufferIdxToQueue(queuesCount, Memory::MallocType::Std),
        queues(queuesCount, Memory::MallocType::Std),
        bestsCosts(problem->variables.getCapacity(), Memory::MallocType::Std)
    {
        unsigned int const storageSize =  StateType::sizeOfStorage(problem);
        std::byte* const storages = StateType::mallocStorages(problem, this->capacity, Memory::MallocType::Std);
        for(unsigned int stateIdx = 0; stateIdx < this->capacity; stateIdx += 1)
        {
            new (this->at(stateIdx)) StateType(problem, &storages[storageSize * stateIdx]);
        }

        for (std::unordered_map<std::size_t, unsigned int>* lvlBestsCosts = bestsCosts.begin(); lvlBestsCosts != bestsCosts.end(); lvlBestsCosts += 1)
        {
            new (lvlBestsCosts) std::unordered_map<std::size_t, unsigned int>();
        }
    }

    template<typename StateType>
    bool PriorityQueuesManager<StateType>::areQueuesEmpty()
    {
       return this->isEmpty();
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
    unsigned int PriorityQueuesManager<StateType>::getQueuesSize()
    {
        return this->getSize();
    }

    template<typename StateType>
    void PriorityQueuesManager<StateType>::reduceQueuesByClass()
    {
        assert(not queues.isEmpty());
        MaxHeap<QueuedState<StateType>>* queue = *queues.front();

        for(QueuedState<StateType>* queuedState = queue->begin(); queuedState != queue->end(); queuedState += 1)
        {
            unsigned int lvl = queuedState->state->selectedValues.getSize();
            std::size_t key = hash(queuedState->state);
            unsigned int stateCost = queuedState->state->cost;
            if (bestsCosts.at(lvl)->count(key) == 0)
            {
                bestsCosts.at(lvl)->insert({key, stateCost});
            }
            else
            {
                unsigned int bestCost = bestsCosts.at(lvl)->at(key);
                if (stateCost < bestCost)
                {
                    bestsCosts.at(lvl)->at(key) = stateCost;
                }
            }
        }

        Vector<StateType*> toRemove(this->getSize(), Memory::MallocType::Std);
        for (QueuedState<StateType>* queuedState = queue->begin(); queuedState != queue->end(); queuedState += 1)
        {
            unsigned int lvl = queuedState->state->selectedValues.getSize();
            std::size_t key = hash(queuedState->state);
            unsigned int stateCost = queuedState->state->cost;
            if (bestsCosts.at(lvl)->at(key) < stateCost)
            {
                toRemove.incrementSize();
                *toRemove.back() = const_cast<StateType*>(queuedState->state);
            }
        }

        for (StateType** stateToRemove = toRemove.begin(); stateToRemove != toRemove.end(); stateToRemove += 1)
        {
            this->dequeue(*stateToRemove);
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

    template<typename StateType>
    std::size_t PriorityQueuesManager<StateType>::hash(StateType const * state)
    {
        static_assert(std::is_same_v<decltype(state->selectedValues), LightVector<uint8_t>>);

        std::hash<std::bitset<UINT8_MAX>> bitsetHashFn;

        std::bitset<UINT8_MAX> bitset;
        for(unsigned int i = 0; i < state->selectedValues.getSize(); i += 1)
        {
            bitset.set(*state->selectedValues.at(i));
        }
        size_t hash = bitsetHashFn(bitset);
        hash *= *state->selectedValues.back();

        return hash;
    }
}