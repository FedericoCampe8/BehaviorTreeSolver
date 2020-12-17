#pragma once

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/swap.h>

#include "../DP/State.cuh"
#include "../Misc/Helpers.cuh"
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
            T const model;

        public:
            MDD(OP::Problem const * problem, unsigned int width);
            __host__ __device__ void buildTopDown(Type type, DP::State* top, unsigned int& cutsetSize, DP::State* cutsetBuffer, DP::State* bottom, std::byte* scratchpadMem);
        private:
            static unsigned int calcFanout(OP::Problem const * problem);
            unsigned int calcScratchpadMemSize() const;

    };

    template<typename T>
    MDD<T>::MDD(OP::Problem const * problem, unsigned int width) :
        width(width),
        fanout(calcFanout(problem)),
        model(problem),
        scratchpadMemSize(calcScratchpadMemSize())
    {}


    template<typename T>
    __host__ __device__
    void MDD<T>::buildTopDown(Type type, DP::State* top, unsigned int& cutsetSize, DP::State* cutsetBuffer, DP::State* bottom, std::byte* scratchpadMem)
    {
        unsigned int const variablesCount = model->problem->variables.getCapacity();

        // Current states buffer
        T::StateType* rawArrayOfStates = Misc::getRawArrayOfStates<T::StateType>(variablesCount, width, scratchpadMem);
        RuntimeArray<T::StateType> currentStatesBuffer(width, reinterpret_cast<std::byte*>(rawArrayOfStates));

        // Next states buffer
        rawArrayOfStates = Misc::getRawArrayOfStates<T::StateType>(variablesCount, width, currentStatesBuffer.storageEnd());
        RuntimeArray<T::StateType> nextStatesBuffer(width, reinterpret_cast<std::byte*>(rawArrayOfStates));

        // Auxiliary information
        RuntimeArray<uint32_t> costs(fanout * width, Memory::align(4, nextStatesBuffer.storageEnd()));
        RuntimeArray<uint8_t> indices(fanout * width, Memory::align(4, costs.storageEnd()));

        assert(indices.storageEnd() < scratchpadMem + scratchpadMemSize);

        // Root
        currentStatesBuffer[0] = *reinterpret_cast<T::StateType>(top);

        // Build
        bool cutsetInitialized = false;
        unsigned int currentStatesCount = 1;
        unsigned int nextStatesCount = 0;
        for(unsigned int variableIdx = top->selectedValues.getSize(); variableIdx <variablesCount; variableIdx += 1)
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
                thrust::for_each(thrust::seq, indices.begin(), indices.begin() + costsCount, [=] (uint8_t& index)
                {
                    unsigned int const currentStateIdx = index / fanout;
                    unsigned int const edgeIdx = index % fanout;
                    unsigned int const selectedValue = model->problem->variables[variableIdx].minValue + edgeIdx;
                    unsigned int const cutsetStateIdx = thrust::distance(indices.begin(), &index);
                    model.makeState(&currentStatesBuffer[currentStateIdx], selectedValue, costs[cutsetStateIdx], &cutsetBuffer[cutsetStateIdx]);
                });

                cutsetSize = costsCount;
                cutsetInitialized = true;
            }

            // Add next states
            assert(nextStatesCount <= indices.getCapacity());
            thrust::for_each(thrust::seq, indices.begin(), indices.begin() + costsCount, [=] (uint8_t& index)
            {
                unsigned int const currentStateIdx = index / fanout;
                unsigned int const edgeIdx =  index % fanout;
                unsigned int const selectedValue = model->problem->variables[variableIdx].minValue + edgeIdx;
                unsigned int const nextStateIdx = thrust::distance(indices.begin(), &index);
                if(nextStateIdx < nextStatesCount)
                {
                    model.makeState(&currentStatesBuffer[currentStateIdx], selectedValue, costs[nextStateIdx], &nextStatesBuffer[nextStateIdx]);
                }
                else if (type == Type::Relaxed)
                {
                    model.mergeNextState(&currentStatesBuffer[currentStateIdx], selectedValue, &nextStatesBuffer[nextStatesCount - 1]);
                }
            });

            //Prepare for the next loop
            thrust::swap(currentStatesBuffer, nextStatesBuffer);
            currentStatesCount = nextStatesCount;
            nextStatesCount = 0;
        }

        //Copy bottom
        bottom = *currentStatesBuffer[0];
    }

    template<typename T>
    unsigned int MDD<T>::calcFanout(OP::Problem const * problem)
    {
        return thrust::transform_reduce(thrust::seq, problem->variables.begin(), problem->variables.end(), OP::Variable::cardinality, 0, thrust::maximum<unsigned int>());
    }

    template<typename T>
    __host__ __device__
    unsigned int MDD<T>::calcScratchpadMemSize() const
    {
        unsigned int const variablesCount = model->problem->variables.getCapacity();

        return
            Misc::getSizeOfRawArrayOfStates(variablesCount, width) * 2 + // currentStatesBuffer, nextStatesBuffer
            sizeof(uint32_t) * width * fanout + // costs
            sizeof(uint8_t) * width * fanout; // indices
    }
}