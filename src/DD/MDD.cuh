#pragma once

#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/swap.h>

#include "../DP/State.cuh"
#include "../OP/Problem.cuh"
#include "../LNS/Neighbourhood.cuh"

namespace DD
{
    template<typename ModelType, typename ProblemType, typename StateType>
    class MDD
    {

        // Aliases, Enums, ...
        public:
            enum Type {Relaxed, Restricted};

        // Members
        public:
            ModelType const * model;
            unsigned int width;
            unsigned int maxValue;
            unsigned int scratchpadMemSize;

        // Functions
        public:
            MDD(ModelType const * model, unsigned int width);
            __host__ __device__ void buildTopDown(Type type, StateType const * top, LightVector<StateType>* cutset, StateType* bottom, LNS::Neighbourhood const * neighbourhood, std::byte* scratchpadMem) const;
            unsigned int calcCutsetMaxSize() const;
        private:
            unsigned int calcScratchpadMemSize() const;

    };

    template<typename ModelType, typename ProblemType, typename StateType>
    MDD<ModelType,ProblemType,StateType>::MDD(ModelType const * model, unsigned int width) :
        model(model),
        width(width),
        maxValue(model->problem->calcMaxValue()),
        scratchpadMemSize(calcScratchpadMemSize())
    {}

    template<typename ModelType, typename ProblemType, typename StateType>
    __host__ __device__
    void MDD<ModelType,ProblemType,StateType>::buildTopDown(Type type, StateType const * top, LightVector<StateType>* cutset, StateType* bottom, LNS::Neighbourhood const * neighbourhood, std::byte* scratchpadMem) const
    {
        std::byte* freeScratchpadMem = scratchpadMem;

        // Current states
        LightVector<StateType> currentStates(width, reinterpret_cast<StateType*>(freeScratchpadMem));
        freeScratchpadMem = Memory::align(8u, currentStates.LightArray<StateType>::end());
        unsigned int const storageSize = StateType::sizeOfStorage(model->problem);
        for(unsigned int stateIdx = 0; stateIdx < currentStates.getCapacity(); stateIdx += 1)
        {
            new (currentStates.LightArray<StateType>::at(stateIdx)) StateType(model->problem, freeScratchpadMem);
            freeScratchpadMem += storageSize;
        }
        freeScratchpadMem = Memory::align(8u, freeScratchpadMem);

        // Next states
        LightVector<StateType> nextStates(width, reinterpret_cast<StateType*>(freeScratchpadMem));
        freeScratchpadMem = Memory::align(8u, nextStates.LightArray<StateType>::end());
        for(unsigned int stateIdx = 0; stateIdx < nextStates.getCapacity(); stateIdx += 1)
        {
            new (nextStates.LightArray<StateType>::at(stateIdx)) StateType(model->problem, freeScratchpadMem);
            freeScratchpadMem += storageSize;
        }
        freeScratchpadMem = Memory::align(8u, freeScratchpadMem);

        // Auxiliary information
        LightVector<DP::CostType> costs(maxValue * width, reinterpret_cast<DP::CostType*>(freeScratchpadMem));
        freeScratchpadMem = Memory::align(8u, costs.LightArray<DP::CostType>::end());

        assert(width * maxValue < UINT32_MAX);
        LightVector<uint32_t> indices(maxValue * width, reinterpret_cast<uint32_t*>(freeScratchpadMem));
        freeScratchpadMem = Memory::align(8u, indices.LightArray<uint32_t>::end());

        assert(freeScratchpadMem < scratchpadMem + scratchpadMemSize);

        // Root
        currentStates.pushBack(top);

        // Build
        bool cutsetInitialized = false;
        unsigned int const variablesCount = model->problem->variables.getCapacity();
        for(unsigned int variableIdx = top->selectedValues.getSize(); variableIdx < variablesCount; variableIdx += 1)
        {
            // Initialize indices
            indices.resize(maxValue * currentStates.getSize());
            thrust::sequence(thrust::seq, indices.begin(), indices.end());

            // Initialize costs
            costs.resize(maxValue * currentStates.getSize());
            thrust::fill(thrust::seq, costs.begin(), costs.end(), DP::MaxCost);

            // Calculate costs
            for (unsigned int currentStateIdx = 0; currentStateIdx < currentStates.getSize(); currentStateIdx += 1)
            {
                model->calcCosts(variableIdx, currentStates[currentStateIdx], neighbourhood, costs[maxValue * currentStateIdx]);
            }

            // Sort indices by costs
            thrust::sort_by_key(thrust::seq, costs.begin(), costs.end(), indices.begin());

            // Discards bad egdes by cost
            uint32_t* const costsEnd = thrust::lower_bound(thrust::seq, costs.begin(), costs.end(), DP::MaxCost);
            if (costsEnd != costs.end())
            {
                unsigned int size = costs.indexOf(costsEnd);
                if (size > 0)
                {
                    costs.resize(size);
                    indices.resize(size);
                }
                else
                {
                    *bottom = *top;
                    bottom->cost = DP::MaxCost;
                    return;
                }
            }

            if (variableIdx < variablesCount - 1)
            {
                nextStates.resize(min(width, indices.getSize()));
            }
            else
            {
                nextStates.resize(1);
            }

            // Save cutset
            if(type == Type::Relaxed and (not cutsetInitialized))
            {
                cutset->resize(indices.getSize());
                for(unsigned int cutsetStateIdx = 0; cutsetStateIdx < indices.getSize(); cutsetStateIdx += 1)
                {
                    uint32_t const index = *indices[cutsetStateIdx];
                    unsigned int const currentStateIdx = index / maxValue;
                    unsigned int const edgeIdx = index % maxValue;
                    OP::ValueType const value = model->problem->variables[variableIdx]->minValue + edgeIdx;
                    model->makeState(currentStates[currentStateIdx], value, *costs[cutsetStateIdx], cutset->at(cutsetStateIdx));
                };

                cutsetInitialized = true;
            }

            // Add next states
            for(unsigned int nextStateIdx = 0; nextStateIdx < indices.getSize(); nextStateIdx += 1)
            {
                uint32_t const index = *indices[nextStateIdx];
                unsigned int const currentStateIdx = index / maxValue;
                unsigned int const edgeIdx = index % maxValue;
                OP::ValueType const value = model->problem->variables[variableIdx]->minValue + edgeIdx;
                if (nextStateIdx < nextStates.getSize())
                {
                    model->makeState(currentStates[currentStateIdx], value, *costs[nextStateIdx], nextStates[nextStateIdx]);
                }
                else if (type == Type::Relaxed)
                {
                    model->mergeState(currentStates[currentStateIdx], value, nextStates.back());
                }
            }

            //Prepare for the next loop
            LightVector<StateType>::swap(currentStates, nextStates);
        }

        //Copy bottom
        *bottom = *currentStates[0];
    }

    template<typename ModelType, typename ProblemType, typename StateType>
    unsigned int MDD<ModelType, ProblemType, StateType>::calcCutsetMaxSize() const
    {
        return width * maxValue;
    }

    template<typename ModelType, typename ProblemType, typename StateType>
    unsigned int MDD<ModelType,ProblemType,StateType>::calcScratchpadMemSize() const
    {
        unsigned int const stateSize = sizeof(StateType);
        unsigned int const stateStorageSize = StateType::sizeOfStorage(model->problem);
        unsigned int const cutsetMaxSize = calcCutsetMaxSize();

        return
            stateSize * width + // currentStates
            stateStorageSize * width  +
            stateSize * width + // nextStates
            stateStorageSize * width  +
            sizeof(DP::CostType) * cutsetMaxSize + // costs
            sizeof(uint32_t) * cutsetMaxSize + // indices
            8 * 6; // Alignment
    }

}