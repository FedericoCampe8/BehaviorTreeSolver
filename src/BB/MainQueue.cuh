#pragma once

#include <thrust/distance.h>

#include <Containers/Buffer.cuh>
#include <Containers/MaxHeap.cuh>

#include "../DD/MDD.cuh"
#include "QueuedState.cuh"

namespace BB
{
    template<typename T>
    class MainQueue
    {
        private:
            using StateType = T::StateType;
            using QueuedStateType = BB::QueuedState<StateType>;

        private:
            Buffer<StateType> statesBuffer;

            Array<unsigned int> bufferToCpuHeap;
            Array<unsigned int> bufferToGpuHeap;

            MaxHeap<QueuedStateType> cpuHeap;
            MaxHeap<QueuedStateType> gpuHeap;

        public:
            MainQueue(DD::MDD<T> const & mdd, unsigned int queuesSize, MaxHeap::Comparator cpuCmp, MaxHeap::Comparator gpuCmp);
            void dequeue(QueuedStateType const & state);
            void enqueue(unsigned int lowerbound, unsigned int upperbound, StateType const & state);
            unsigned int getCapacity() const;
            unsigned int getSize() const;
            QueuedStateType& getStateForCpu() const;
            QueuedStateType& getStateForGpu() const;
            bool isEmpty();
            bool isFull();
    };

    template<typename T>
    MainQueue<T>::MainQueue(DD::MDD<T> const & mdd, unsigned int queuesSize, MaxHeap::Comparator cpuCmp, MaxHeap::Comparator gpuCmp) :
        statesBuffer(queuesSize, Memory::MallocType::Std),
        bufferToCpuHeap(queuesSize, Memory::MallocType::Std),
        bufferToGpuHeap(queuesSize, Memory::MallocType::Std),
        cpuHeap(queuesSize, cpuCmp, Memory::MallocType::Std),
        gpuHeap(queuesSize, gpuCmp, Memory::MallocType::Std)
    {
        T::ProblemType const & problem = mdd.model.problem;
        unsigned int storageSize =  T::StateType::sizeOfStorage(problem);
        std::byte* storages = T::StateType::mallocStorages(statesBuffer.getCapacity(), problem, Memory::MallocType::Std);
        for(unsigned int stateIdx = 0; stateIdx < statesBuffer.getCapacity(); stateIdx += 1)
        {
            new (&statesBuffer[stateIdx]) T::StateType(problem, &storages[storageSize * stateIdx]);
        }
    }

    template<typename T>
    void MainQueue<T>::dequeue(QueuedStateType const & queuesState)
    {
        unsigned int const bufferIdx = statesBuffer.indexOf(queuesState.state);

        unsigned int const cpuHeapIdx = bufferToCpuHeap[bufferIdx];
        cpuHeap.erase(cpuHeapIdx);

        unsigned int const gpuHeapIdx = bufferToGpuHeap[bufferIdx];
        gpuHeap.erase(gpuHeapIdx);
    }

    template<typename T>
    void MainQueue<T>::enqueue(unsigned int lowerbound, unsigned int upperbound, StateType const & t)
    {
        T& state = statesBuffer.insert(t);
        unsigned int const bufferIdx = statesBuffer.indexOf(state);

        unsigned int * const cpuHeapIdx = &bufferToCpuHeap[bufferIdx];
        *cpuHeapIdx = cpuHeap.getSize();
        QueuedStateType cpuToEnqueue(cpuHeapIdx, lowerbound, upperbound, state);
        cpuHeap.pushBack(cpuToEnqueue);
        cpuHeap.insertBack();

        unsigned int * const gpuHeapIdx = &bufferToGpuHeap[bufferIdx];
        *gpuHeapIdx = gpuHeap.getSize();
        QueuedState<T> gpuToEnqueue(gpuHeapIdx, lowerbound, upperbound, state);
        gpuHeap.pushBack(gpuToEnqueue);
        gpuHeap.insertBack();
    }

    template<typename T>
    unsigned int MainQueue<T>::getCapacity() const
    {
        statesBuffer.getCapacity();
    }

    template<typename T>
    unsigned int MainQueue<T>::getSize() const
    {
        statesBuffer.getSize();
    }

    template<typename T>
    QueuedStateType& MainQueue<T>::getStateForCpu() const
    {
        return cpuHeap.front();
    }

    template<typename T>
    QueuedStateType& MainQueue<T>::getStateForGpu() const
    {
        return gpuHeap.front();
    }

    template<typename T>
    bool MainQueue<T>::isEmpty()
    {
        return statesBuffer.isEmpty();
    }

    template<typename T>
    bool MainQueue<T>::isFull()
    {
        return statesBuffer.isFull();
    }
}