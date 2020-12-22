#pragma once

#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/swap.h>

#include "../DP/State.cuh"
#include "../OP/Problem.cuh"

namespace DD
{
    template<typename ModelType, typename ProblemType, typename StateType>
    class MDD
    {

        //
        public:
            enum Type {Relaxed, Restricted};

        // Members
        public:
            ModelType const * model;
            unsigned int width;
            unsigned int fanout;
            unsigned int cutsetMaxSize;
            unsigned int scratchpadMemSize;

        // Functions
        public:
            MDD(ModelType const * model, unsigned int width);
            __host__ __device__ void buildTopDown(Type type, StateType const * top, LightVector<StateType>* cutset, StateType* bottom, std::byte* scratchpadMem) const;
        private:
            static unsigned int calcFanout(ProblemType const * problem);
            unsigned int calcScratchpadMemSize() const;

    };

    template<typename ModelType, typename ProblemType, typename StateType>
    MDD<ModelType,ProblemType,StateType>::MDD(ModelType const * model, unsigned int width) :
        model(model),
        width(width),
        fanout(calcFanout(model->problem)),
        cutsetMaxSize(width*fanout),
        scratchpadMemSize(calcScratchpadMemSize())
    {}

    template<typename ModelType, typename ProblemType, typename StateType>
    __host__ __device__
    void MDD<ModelType,ProblemType,StateType>::buildTopDown(Type type, StateType const * top, LightVector<StateType>* cutset, StateType* bottom, std::byte* scratchpadMem) const
    {
        std::byte* freeScratchpadMem = scratchpadMem;

        // Current states
        LightArray<StateType> currentStatesBuffer(width, reinterpret_cast<StateType*>(freeScratchpadMem));
        freeScratchpadMem = Memory::align(4, reinterpret_cast<std::byte*>(currentStatesBuffer.end()));
        unsigned int const storageSize = StateType::sizeOfStorage(model->problem);
        for(unsigned int stateIdx = 0; stateIdx < currentStatesBuffer.getCapacity(); stateIdx += 1)
        {
            new (currentStatesBuffer[stateIdx]) StateType(model->problem, freeScratchpadMem);
            freeScratchpadMem += storageSize;
        }
        freeScratchpadMem = Memory::align(4, freeScratchpadMem);

        // Next states
        LightArray<StateType> nextStatesBuffer(width, reinterpret_cast<StateType*>(freeScratchpadMem));
        freeScratchpadMem = Memory::align(4, reinterpret_cast<std::byte*>(nextStatesBuffer.end()));
        for(unsigned int stateIdx = 0; stateIdx < nextStatesBuffer.getCapacity(); stateIdx += 1)
        {
            new (nextStatesBuffer[stateIdx]) StateType(model->problem, freeScratchpadMem);
            freeScratchpadMem += storageSize;
        }
        freeScratchpadMem = Memory::align(4, freeScratchpadMem);

        // Auxiliary information
        LightArray<uint32_t> costs(fanout * width, reinterpret_cast<uint32_t*>(freeScratchpadMem));
        freeScratchpadMem = Memory::align(4, reinterpret_cast<std::byte*>(costs.end()));

        LightArray<uint8_t> indices(fanout * width, reinterpret_cast<uint8_t*>(freeScratchpadMem));
        freeScratchpadMem = Memory::align(4, reinterpret_cast<std::byte*>(indices.end()));

        assert(freeScratchpadMem < scratchpadMem + scratchpadMemSize);

        // Root
        *currentStatesBuffer[0] = *top;

        // Build
        bool cutsetInitialized = false;
        unsigned int currentStatesCount = 1;
        unsigned int nextStatesCount = 0;
        unsigned int const variablesCount = model->problem->variables.getCapacity();
        for(unsigned int variableIdx = top->selectedValues.getSize(); variableIdx < variablesCount; variableIdx += 1)
        {
            // Initialize indices
            thrust::sequence(thrust::seq, indices.begin(), indices.end());

            // Initialize costs
            thrust::fill(costs.begin(), costs.end(), StateType::MaxCost);

            // Calculate costs
            assert(currentStatesCount <= currentStatesBuffer.getCapacity());
            for(unsigned int currentStateIdx = 0; currentStateIdx < currentStatesCount; currentStateIdx += 1)
            {
                model->calcCosts(variableIdx, currentStatesBuffer[currentStateIdx], costs[fanout * currentStateIdx]);
            }

            // Sort indices by costs
            thrust::sort_by_key(thrust::seq, costs.begin(), costs.end(), indices.begin());

            // Count next states
            uint32_t* const costsEnd = thrust::lower_bound(thrust::seq, costs.begin(), costs.end(), StateType::MaxCost);
            unsigned int const costsCount = costs.indexOf(costsEnd);
            nextStatesCount = min(width, costsCount);
            if(variableIdx == variablesCount - 1)
            {
                nextStatesCount = 1;
            }

            // Save cutset
            if(type == Type::Relaxed and (not cutsetInitialized) and costsCount > width)
            {
                cutset->resize(costsCount);

                for(unsigned int cutsetStateIdx = 0; cutsetStateIdx < costsCount; cutsetStateIdx += 1)
                {
                    unsigned int const index = *indices[cutsetStateIdx];
                    unsigned int const currentStateIdx = index / fanout;
                    unsigned int const edgeIdx = index % fanout;
                    unsigned int const selectedValue = model->problem->variables[variableIdx]->minValue + edgeIdx;
                    model->makeState(currentStatesBuffer[currentStateIdx], selectedValue, *costs[cutsetStateIdx], cutset->at(cutsetStateIdx));
                };

                cutsetInitialized = true;
            }

            // Add next states
            assert(nextStatesCount <= indices.getCapacity());
            for(unsigned int nextStateIdx = 0; nextStateIdx < costsCount; nextStateIdx += 1)
            {
                unsigned int const index = *indices[nextStateIdx];
                unsigned int const currentStateIdx = index / fanout;
                unsigned int const edgeIdx =  index % fanout;
                unsigned int const selectedValue = model->problem->variables[variableIdx]->minValue + edgeIdx;
                if(nextStateIdx < nextStatesCount)
                {
                    model->makeState(currentStatesBuffer[currentStateIdx], selectedValue, *costs[nextStateIdx], nextStatesBuffer[nextStateIdx]);
                }
                else if (type == Type::Relaxed)
                {
                    model->mergeState(currentStatesBuffer[currentStateIdx], selectedValue, nextStatesBuffer[nextStatesCount - 1]);
                }
            }

            //Prepare for the next loop
            LightArray<StateType>::swap(&currentStatesBuffer, &nextStatesBuffer);
            currentStatesCount = nextStatesCount;
        }

        //Copy bottom
        *bottom = *currentStatesBuffer[0];
    }

    template<typename ModelType, typename ProblemType, typename StateType>
    unsigned int MDD<ModelType,ProblemType,StateType>::calcFanout(ProblemType const * problem)
    {
        unsigned int fanout = 0;
        for (OP::Variable* variable = problem->variables.begin(); variable != problem->variables.end(); variable += 1)
        {
            fanout = max(fanout, variable->cardinality());
        }

        return fanout;
    }

    template<typename ModelType, typename ProblemType, typename StateType>
    unsigned int MDD<ModelType,ProblemType,StateType>::calcScratchpadMemSize() const
    {
        unsigned int const stateSize = sizeof(StateType);
        unsigned int const stateStorageSize = StateType::sizeOfStorage(model->problem);

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