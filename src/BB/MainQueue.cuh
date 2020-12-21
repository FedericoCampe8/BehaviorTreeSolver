#pragma once

#include <thrust/distance.h>

#include <Containers/Buffer.cuh>
#include <Containers/MaxHeap.cuh>

#include "../DD/MDD.cuh"
#include "QueuedState.cuh"

namespace BB
{
    template<typename ModelType, typename ProblemType, typename StateType>
    class MainQueue
    {
        public:
            typedef bool (*Comparator) (QueuedState<StateType> const & s0, QueuedState<StateType> const & t1);

        private:
            Buffer<StateType> statesBuffer;

            Array<unsigned int> bufferToCpuHeap;
            Array<unsigned int> bufferToGpuHeap;

            MaxHeap<QueuedState<StateType>> cpuHeap;
            MaxHeap<QueuedState<StateType>> gpuHeap;

        public:
            MainQueue(DD::MDD<ModelType,ProblemType,StateType> const & mdd, unsigned int queuesSize, Comparator cpuCmp, Comparator gpuCmp);
            void dequeue(QueuedState<StateType> const & state);
            void enqueue(unsigned int lowerbound, unsigned int upperbound, StateType const & state);
            unsigned int getCapacity() const;
            unsigned int getSize() const;
            QueuedState<StateType>& getStateForCpu() const;
            QueuedState<StateType>& getStateForGpu() const;
            bool isEmpty();
            bool isFull();
    };

    template<typename ModelType, typename ProblemType, typename StateType>
    MainQueue<ModelType,ProblemType,StateType>::MainQueue(DD::MDD<ModelType,ProblemType,StateType> const & mdd, unsigned int queuesSize, Comparator cpuCmp, Comparator gpuCmp) :
        statesBuffer(queuesSize, Memory::MallocType::Std),
        bufferToCpuHeap(queuesSize, Memory::MallocType::Std),
        bufferToGpuHeap(queuesSize, Memory::MallocType::Std),
        cpuHeap(queuesSize, cpuCmp, Memory::MallocType::Std),
        gpuHeap(queuesSize, gpuCmp, Memory::MallocType::Std)
    {
        ProblemType const & problem = mdd.model.problem;
        unsigned int storageSize =  StateType::sizeOfStorage(problem);
        std::byte* storages = StateType::mallocStorages(statesBuffer.getCapacity(), problem, Memory::MallocType::Std);
        for(unsigned int stateIdx = 0; stateIdx < statesBuffer.getCapacity(); stateIdx += 1)
        {
            new (&statesBuffer[stateIdx]) StateType(problem, &storages[storageSize * stateIdx]);
        }
    }

    template<typename ModelType, typename ProblemType, typename StateType>
    void MainQueue<ModelType,ProblemType,StateType>::dequeue(QueuedState<StateType> const & queuesState)
    {
        unsigned int const bufferIdx = statesBuffer.indexOf(queuesState.state);

        unsigned int const cpuHeapIdx = bufferToCpuHeap[bufferIdx];
        cpuHeap.erase(cpuHeapIdx);

        unsigned int const gpuHeapIdx = bufferToGpuHeap[bufferIdx];
        gpuHeap.erase(gpuHeapIdx);
    }

    template<typename ModelType, typename ProblemType, typename StateType>
    void MainQueue<ModelType,ProblemType,StateType>::enqueue(unsigned int lowerbound, unsigned int upperbound, StateType const & state)
    {
        StateType& bufferedState = statesBuffer.insert(state);
        unsigned int const bufferIdx = statesBuffer.indexOf(bufferedState);

        unsigned int * const cpuHeapIdx = &bufferToCpuHeap[bufferIdx];
        *cpuHeapIdx = cpuHeap.getSize();
        QueuedState<StateType> cpuToEnqueue(cpuHeapIdx, lowerbound, upperbound, bufferedState);
        cpuHeap.pushBack(cpuToEnqueue);
        cpuHeap.insertBack();

        unsigned int * const gpuHeapIdx = &bufferToGpuHeap[bufferIdx];
        *gpuHeapIdx = gpuHeap.getSize();
        QueuedState<StateType> gpuToEnqueue(gpuHeapIdx, lowerbound, upperbound, bufferedState);
        gpuHeap.pushBack(gpuToEnqueue);
        gpuHeap.insertBack();
    }

    template<typename ModelType, typename ProblemType, typename StateType>
    unsigned int MainQueue<ModelType,ProblemType,StateType>::getCapacity() const
    {
        return statesBuffer.getCapacity();
    }

    template<typename ModelType, typename ProblemType, typename StateType>
    unsigned int MainQueue<ModelType,ProblemType,StateType>::getSize() const
    {
        return statesBuffer.getSize();
    }

    template<typename ModelType, typename ProblemType, typename StateType>
    QueuedState<StateType>& MainQueue<ModelType,ProblemType,StateType>::getStateForCpu() const
    {
        return cpuHeap.front();
    }

    template<typename ModelType, typename ProblemType, typename StateType>
    QueuedState<StateType>& MainQueue<ModelType,ProblemType,StateType>::getStateForGpu() const
    {
        return gpuHeap.front();
    }

    template<typename ModelType, typename ProblemType, typename StateType>
    bool MainQueue<ModelType,ProblemType,StateType>::isEmpty()
    {
        return statesBuffer.isEmpty();
    }

    template<typename ModelType, typename ProblemType, typename StateType>
    bool MainQueue<ModelType,ProblemType,StateType>::isFull()
    {
        return statesBuffer.isFull();
    }
}