#pragma once

#include <unordered_map>
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
            Array<std::unordered_map<unsigned int, unsigned int>> bestsCosts;

        // Functions
        public:
            template<typename ProblemType>
            PriorityQueuesManager(ProblemType const * problem, unsigned int queuesSize, unsigned int queuesCount);
            bool areQueuesEmpty();
            void dequeue(StateType const * state);
            void enqueue(StateMetadata<StateType> const * stateMetadata);
            unsigned int getQueuesSize();
            void reduceQueuesByClass();
            void reduceQueuesByCurrentBest(StateType const * bestSolution);
            void reduceQueuesByCostProb(StateType const * bestSolution);
            void registerQueue(MaxHeap<QueuedState<StateType>>* queue);

        private:
            static unsigned int hash(StateType const * state);
    };

    template<typename StateType>
    template<typename ProblemType>
    PriorityQueuesManager<StateType>::PriorityQueuesManager(ProblemType const * problem, unsigned int queuesSize, unsigned int queuesCount) :
        Buffer<StateType>(queuesSize, Memory::MallocType::Std),
        bufferIdxToQueue(queuesCount, Memory::MallocType::Std),
        queues(queuesCount, Memory::MallocType::Std),
        bestsCosts(problem->variables.getCapacity() + 1, Memory::MallocType::Std)
    {
        unsigned int const storageSize =  StateType::sizeOfStorage(problem);
        std::byte* const storages = StateType::mallocStorages(problem, this->capacity, Memory::MallocType::Std);
        for(unsigned int stateIdx = 0; stateIdx < this->capacity; stateIdx += 1)
        {
            new (this->at(stateIdx)) StateType(problem, &storages[storageSize * stateIdx]);
        }

        for (std::unordered_map<unsigned int, unsigned int>* lvlBestsCosts = bestsCosts.begin(); lvlBestsCosts != bestsCosts.end(); lvlBestsCosts += 1)
        {
            new (lvlBestsCosts) std::unordered_map<unsigned int, unsigned int>();
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
            unsigned int key = hash(queuedState->state);
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
            unsigned int key = hash(queuedState->state);
            unsigned int stateCost = queuedState->state->cost;
            //if (bestsCosts.at(lvl)->at(key) < stateCost)
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
    void PriorityQueuesManager<StateType>::reduceQueuesByCurrentBest(StateType const * bestSolution)
    {
        assert(not queues.isEmpty());
        MaxHeap<QueuedState<StateType>>* queue = *queues.front();

        Vector<StateType*> toRemove(this->getSize(), Memory::MallocType::Std);

        double similarity = 0.5;
        while (toRemove.getSize() < this->getSize() * 9 / 10)
        {
            similarity -= 0.05;
            toRemove.clear();

            for (QueuedState<StateType>* queuedState = queue->begin(); queuedState != queue->end(); queuedState += 1)
            {
                double stateSimilarity = 0;
                for (unsigned int i = 0; i < queuedState->state->selectedValues.getSize(); i += 1)
                {
                    if(*queuedState->state->selectedValues.at(i) == *bestSolution->selectedValues.at(i))
                    {
                        stateSimilarity += 1;
                    }
                }

                if (stateSimilarity / static_cast<double>(queuedState->state->selectedValues.getSize()) >= similarity)
                {
                    toRemove.incrementSize();
                    *toRemove.back() = const_cast<StateType*>(queuedState->state);
                }
            }
        }

        for (StateType** stateToRemove = toRemove.begin(); stateToRemove != toRemove.end(); stateToRemove += 1)
        {
            this->dequeue(*stateToRemove);
        }
    }

    template<typename StateType>
    void PriorityQueuesManager<StateType>::reduceQueuesByCostProb(StateType const * bestSolution)
    {
        assert(not queues.isEmpty());
        MaxHeap<QueuedState<StateType>>* queue = *queues.front();

        Vector<StateType*> toRemove(this->getSize(), Memory::MallocType::Std);

        double alpha = 1.00;
        while (toRemove.getSize() < this->getSize() / 2)
        {
            toRemove.clear();
            alpha += 0.01;
            for (QueuedState<StateType>* queuedState = queue->begin(); queuedState != queue->end(); queuedState += 1)
            {
                double avgCostQueuedState = static_cast<double>(queuedState->state->cost) / static_cast<double>(queuedState->state->selectedValues.getSize());
                double finalCostQueuedState = avgCostQueuedState * static_cast<double>(bestSolution->selectedValues.getSize());
                if (finalCostQueuedState > alpha * static_cast<double>(bestSolution->cost))
                {
                    toRemove.incrementSize();
                    *toRemove.back() = const_cast<StateType*>(queuedState->state);
                }
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
    unsigned int PriorityQueuesManager<StateType>::hash(StateType const * state)
    {
        unsigned int hash = thrust::reduce(thrust::seq, state->selectedValues.begin(), state->selectedValues.end(), 0);
        hash *= *state->selectedValues.back();
        return hash;
    }
}