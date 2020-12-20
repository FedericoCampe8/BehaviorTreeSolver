#pragma once

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/swap.h>

#include "../DP/State.cuh"
#include "../OP/Problem.cuh"

namespace DD
{
    template<typename T>
    class MDD
    {
        public:
            enum Type {Relaxed, Restricted};

        public:
            unsigned int width;
            unsigned int fanout;
            unsigned int scratchpadMemSize;
            T const & model;

        public:
            MDD(T const & model, unsigned int width);
            __host__ __device__ void buildTopDown(Type type, T::StateType& top, Vector<T::StateType>& cutset, DP::State& bottom, std::byte* scratchpadMem) const;
        private:
            static unsigned int calcFanout(T::ProblemType const & problem);
            unsigned int calcScratchpadMemSize() const;

    };

    template<typename T>
    MDD<T>::MDD(T const & model, unsigned int width) :
        width(width),
        fanout(calcFanout(model.problem)),
        model(model),
        scratchpadMemSize(calcScratchpadMemSize())
    {}


    template<typename T>
    __host__ __device__
    void MDD<T>::buildTopDown(Type type, T::StateType& top, Vector<T::StateType>& cutset, DP::State& bottom, std::byte* scratchpadMem) const
    {
        std::byte* freeScratchpadMem = scratchpadMem;

        // Current states currentStatesBuffer
        Array<T::StateType> currentStatesBuffer(width, freeScratchpadMem);
        freeScratchpadMem = Memory::align(4, currentStatesBuffer.storageEnd());

        unsigned int storageSize = T::StateType::sizeOfStorage(model.problem);
        for(unsigned int stateIdx = 0; stateIdx < currentStatesBuffer.getCapacity(); stateIdx += 1)
        {
            new (&currentStatesBuffer[stateIdx]) T::StateType(model.problem, freeScratchpadMem);
            freeScratchpadMem += storageSize;
        }
        freeScratchpadMem = Memory::align(4, freeScratchpadMem);

        // Next states currentStatesBuffer
        Array<T::StateType> nextStatesBuffer(width, freeScratchpadMem);
        freeScratchpadMem = Memory::align(4, nextStatesBuffer.storageEnd());
        for(unsigned int stateIdx = 0; stateIdx < nextStatesBuffer.getCapacity(); stateIdx += 1)
        {
            new (&nextStatesBuffer[stateIdx]) T::StateType(model.problem, freeScratchpadMem);
            freeScratchpadMem += storageSize;
        }
        freeScratchpadMem = Memory::align(4, freeScratchpadMem);

        // Auxiliary information
        Array<uint32_t> costs(fanout * width, freeScratchpadMem);
        Array<uint8_t> indices(fanout * width, costs.storageEnd());

        assert(indices.storageEnd() < scratchpadMem + scratchpadMemSize);

        // Root
        currentStatesBuffer[0] = *reinterpret_cast<T::StateType>(top);

        // Build
        bool cutsetInitialized = false;
        unsigned int currentStatesCount = 1;
        unsigned int nextStatesCount = 0;
        unsigned int const variablesCount = model.problem.variables.getCapacity();
        for(unsigned int variableIdx = top->selectedValues.getSize(); variableIdx < variablesCount; variableIdx += 1)
        {
            // Initialize indices
            thrust::sequence(thrust::seq, indices.begin(), indices.end());

            // Initialize costs
            thrust::fill(costs.begin(), costs.end(), T::StateType::MaxCost);

            // Calculate costs
            assert(currentStatesCount <= currentStatesBuffer.getCapacity());
            for(unsigned int currentStateIdx = 0; currentStateIdx < currentStatesCount; currentStateIdx += 1)
            {
                model.calcCosts(variableIdx, &currentStatesBuffer[currentStateIdx], &costs[fanout * currentStateIdx]);
            }

            // Sort indices by costs
            thrust::sort_by_key(thrust::seq, costs.begin(), costs.end(), indices.begin());

            // Count next states
            uint32_t* const costsEnd = thrust::lower_bound(thrust::seq, costs.begin(), costs.end(), T::StateType::MaxCost);
            unsigned int const costsCount = thrust::distance(costs.begin(), costsEnd);
            nextStatesCount = min(width, costsCount);
            if(variableIdx == variablesCount - 1)
            {
                nextStatesCount = 1;
            }

            // Save cutset
            if(type == Type::Relaxed and (not cutsetInitialized) and costsCount > width)
            {
                cutset.resize(costsCount);

                for(unsigned int cutsetStateIdx = 0; cutsetStateIdx < costsCount; cutsetStateIdx += 1)
                {
                    unsigned int const index = indices[cutsetStateIdx];
                    unsigned int const currentStateIdx = index / fanout;
                    unsigned int const edgeIdx = index % fanout;
                    unsigned int const selectedValue = model.problem.variables[variableIdx].minValue + edgeIdx;
                    model.makeState(&currentStatesBuffer[currentStateIdx], selectedValue, costs[cutsetStateIdx], &cutset[cutsetStateIdx]);
                };

                cutsetInitialized = true;
            }

            // Add next states
            assert(nextStatesCount <= indices.getCapacity());
            for(unsigned int nextStateIdx = 0; nextStateIdx < costsCount; nextStateIdx += 1)
            {
                unsigned int const index = indices[nextStateIdx];
                unsigned int const currentStateIdx = index / fanout;
                unsigned int const edgeIdx =  index % fanout;
                unsigned int const selectedValue = model.problem.variables[variableIdx].minValue + edgeIdx;
                if(nextStateIdx < nextStatesCount)
                {
                    model.makeState(&currentStatesBuffer[currentStateIdx], selectedValue, costs[nextStateIdx], &nextStatesBuffer[nextStateIdx]);
                }
                else if (type == Type::Relaxed)
                {
                    model.mergeState(&currentStatesBuffer[currentStateIdx], selectedValue, &nextStatesBuffer[nextStatesCount - 1]);
                }
            }

            //Prepare for the next loop
            Array<T::StateType>::swap(currentStatesBuffer, nextStatesBuffer);
            currentStatesCount = nextStatesCount;
        }

        //Copy bottom
        bottom = *currentStatesBuffer[0];
    }

    template<typename T>
    unsigned int MDD<T>::calcFanout(T::ProblemType const & problem)
    {
        return thrust::transform_reduce(thrust::seq, problem.variables.begin(), problem.variables.end(), OP::Variable::cardinality, 0, thrust::maximum<unsigned int>());
    }

    template<typename T>
    __host__ __device__
    unsigned int MDD<T>::calcScratchpadMemSize() const
    {
        unsigned int const variablesCount = model.problem.variables.getCapacity();
        unsigned int const stateSize = sizeof(T::StateType);
        unsigned int const stateStorageSize = T::StateType::sizeOfStorage(variablesCount);

        return
            stateSize * width + // Current states
            stateStorageSize * width  +
            stateSize * width + // Next states
            stateStorageSize * width  +
            sizeof(uint32_t) * width * fanout + // Costs
            sizeof(uint8_t) * width * fanout + // Indices
            4 * 6; // Alignment
    }
}