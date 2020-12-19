#pragma once

#include <thrust/distance.h>

#include <Containers/StaticSet.cuh>
#include <Containers/StaticMaxHeap.cuh>

#include "AugmentedState.cuh"
namespace BB
{
    template<typename T>
    class QueuesManager
    {
        public:
            StaticSet<T> buffer;

            RuntimeArray<unsigned int> cpuBufferToHeap;
            RuntimeArray<unsigned int> gpuBufferToHeap;

            StaticMaxHeap<AugmentedState<T>> cpuHeap;
            StaticMaxHeap<AugmentedState<T>> gpuHeap;

        public:
            QueuesManager(unsigned int queuesSize, StaticMaxHeap::Comparator cpuCmp, StaticMaxHeap::Comparator gpuCmp);
            void enqueue(unsigned int lowerbound, unsigned int upperbound, T const & t);
            void dequeue(T const * t);
    };

    template<typename T>
    QueuesManager<T>::QueuesManager(unsigned int queuesSize, StaticMaxHeap::Comparator cpuCmp, StaticMaxHeap::Comparator gpuCmp) :
        buffer(queuesSize, Memory::MallocType::Std),
        cpuBufferToHeap(queuesSize, Memory::MallocType::Std),
        gpuBufferToHeap(queuesSize, Memory::MallocType::Std),
        cpuHeap(queuesSize, cpuCmp, Memory::MallocType::Std),
        gpuHeap(queuesSize, gpuCmp, Memory::MallocType::Std)
    {}

    template<typename T>
    void QueuesManager<T>::enqueue(unsigned int lowerbound, unsigned int upperbound, T const & t)
    {
        T* buffered = buffer.insert(t);
        unsigned int const bufferIdx = buffer.indexOf(buffered);

        unsigned int * const cpuHeapIdx = &cpuBufferToHeap[bufferIdx];
        *cpuHeapIdx = cpuHeap.vector.getSize();
        AugmentedState<T> cpuToEnqueue(bufferIdx, cpuHeapIdx, lowerbound, upperbound, buffered);
        cpuHeap.vector.pushBack(cpuToEnqueue);
        cpuHeap.insertBack();

        unsigned int * const gpuHeapIdx = &gpuBufferToHeap[bufferIdx];
        *gpuHeapIdx = gpuHeap.vector.getSize();
        AugmentedState<T> gpuToEnqueue(bufferIdx, gpuHeapIdx, lowerbound, upperbound, buffered);
        gpuHeap.vector.pushBack(gpuToEnqueue);
        gpuHeap.insertBack();
    }

    template<typename T>
    void QueuesManager<T>::dequeue(T const * t)
    {
        unsigned int const bufferIdx = buffer.indexOf(t);

        unsigned int const cpuHeapIdx = cpuBufferToHeap[bufferIdx];
        cpuHeap.eraseIndex(cpuHeapIdx);

        unsigned int const gpuHeapIdx = gpuBufferToHeap[bufferIdx];
        gpuHeap.eraseIndex(gpuHeapIdx);
    }
}