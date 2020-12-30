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

        // Aliases, Enums, ...
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
            void printStates(LightVector<StateType> const * states) const;

    };

    template<typename ModelType, typename ProblemType, typename StateType>
    MDD<ModelType,ProblemType,StateType>::MDD(ModelType const * model, unsigned int width) :
        model(model),
        width(width),
        fanout(calcFanout(model->problem)),
        cutsetMaxSize(width * fanout),
        scratchpadMemSize(calcScratchpadMemSize())
    {}

    template<typename ModelType, typename ProblemType, typename StateType>
    __host__ __device__
    void MDD<ModelType,ProblemType,StateType>::buildTopDown(Type type, StateType const * top, LightVector<StateType>* cutset, StateType* bottom, std::byte* scratchpadMem) const
    {
        std::byte* freeScratchpadMem = scratchpadMem;

        // Current states
        LightVector<StateType> currentStates(width, reinterpret_cast<StateType*>(freeScratchpadMem));
        freeScratchpadMem = Memory::align(8, reinterpret_cast<std::byte*>(currentStates.LightArray<StateType>::end()));
        unsigned int const storageSize = StateType::sizeOfStorage(model->problem);
        for(unsigned int stateIdx = 0; stateIdx < currentStates.getCapacity(); stateIdx += 1)
        {
            new (currentStates.LightArray<StateType>::at(stateIdx)) StateType(model->problem, freeScratchpadMem);
            freeScratchpadMem += storageSize;
        }
        freeScratchpadMem = Memory::align(8, freeScratchpadMem);

        // Next states
        LightVector<StateType> nextStates(width, reinterpret_cast<StateType*>(freeScratchpadMem));
        freeScratchpadMem = Memory::align(8, reinterpret_cast<std::byte*>(nextStates.LightArray<StateType>::end()));
        for(unsigned int stateIdx = 0; stateIdx < nextStates.getCapacity(); stateIdx += 1)
        {
            new (nextStates.LightArray<StateType>::at(stateIdx)) StateType(model->problem, freeScratchpadMem);
            freeScratchpadMem += storageSize;
        }
        freeScratchpadMem = Memory::align(8, freeScratchpadMem);

        // Auxiliary information
        LightVector<uint32_t> costs(fanout * width, reinterpret_cast<uint32_t*>(freeScratchpadMem));
        freeScratchpadMem = Memory::align(8, reinterpret_cast<std::byte*>(costs.LightArray<uint32_t>::end()));

        assert(width * fanout < UINT32_MAX);
        LightVector<uint32_t> indices(fanout * width, reinterpret_cast<uint32_t*>(freeScratchpadMem));
        freeScratchpadMem = Memory::align(8, reinterpret_cast<std::byte*>(indices.LightArray<uint32_t>::end()));

        assert(freeScratchpadMem < scratchpadMem + scratchpadMemSize);

        // Root
        currentStates.incrementSize();
        *currentStates.back() = *top;

        // Build
        bool cutsetInitialized = false;
        unsigned int const variablesCount = model->problem->variables.getCapacity();
        for(unsigned int variableIdx = top->selectedValues.getSize(); variableIdx < variablesCount; variableIdx += 1)
        {
            // Initialize indices
            indices.resize(fanout * currentStates.getSize());
            thrust::sequence(thrust::seq, indices.begin(), indices.end());

            // Initialize costs
            costs.resize(fanout * currentStates.getSize());
            thrust::fill(thrust::seq, costs.begin(), costs.end(), StateType::MaxCost);

            // Calculate costs
            for (unsigned int currentStateIdx = 0; currentStateIdx < currentStates.getSize(); currentStateIdx += 1)
            {
                //currentStates[currentStateIdx]->print();
                model->calcCosts(variableIdx, currentStates[currentStateIdx], costs[fanout * currentStateIdx]);
            }

            // Sort indices by costs
            thrust::sort_by_key(thrust::seq, costs.begin(), costs.end(), indices.begin());

            // Discards bad egdes by cost
            uint32_t* const costsEnd = thrust::lower_bound(thrust::seq, costs.begin(), costs.end(), StateType::MaxCost);
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
                    bottom->cost = StateType::MaxCost;
                    return;
                }
            }

            if (variableIdx < variablesCount - 1)
            {
                nextStates.resize(min(width, static_cast<unsigned int>(indices.getSize())));
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
                    unsigned int const index = *indices[cutsetStateIdx];
                    unsigned int const currentStateIdx = index / fanout;
                    unsigned int const edgeIdx = index % fanout;
                    unsigned int const selectedValue = model->problem->variables[variableIdx]->minValue + edgeIdx;
                    model->makeState(currentStates[currentStateIdx], selectedValue, *costs[cutsetStateIdx], cutset->at(cutsetStateIdx));
                    assert(currentStates[currentStateIdx]->selectedValues.getSize() + 1 == cutset->at(cutsetStateIdx)->selectedValues.getSize());
                };

                cutsetInitialized = true;
            }

            // Add next states
            for(unsigned int nextStateIdx = 0; nextStateIdx < indices.getSize(); nextStateIdx += 1)
            {
                unsigned int const index = *indices[nextStateIdx];
                unsigned int const currentStateIdx = index / fanout;
                unsigned int const edgeIdx = index % fanout;
                unsigned int const selectedValue = model->problem->variables[variableIdx]->minValue + edgeIdx;
                if (nextStateIdx < nextStates.getSize())
                {
                    model->makeState(currentStates[currentStateIdx], selectedValue, *costs[nextStateIdx], nextStates[nextStateIdx]);
                    assert(currentStates[currentStateIdx]->selectedValues.getSize() + 1 == nextStates[nextStateIdx]->selectedValues.getSize());
                }
                else if (type == Type::Relaxed)
                {
                    model->mergeState(currentStates[currentStateIdx], selectedValue, nextStates.back());
                    assert(currentStates[currentStateIdx]->selectedValues.getSize() + 1 == nextStates.back()->selectedValues.getSize());
                }
            }

            for(unsigned int stateIdx = 0; stateIdx < nextStates.getSize(); stateIdx += 1)
            {
                assert(currentStates[0]->selectedValues.getSize() + 1 == nextStates[stateIdx]->selectedValues.getSize());
            }

            //Prepare for the next loop
            LightVector<StateType>::swap(&currentStates, &nextStates);
        }

        //Copy bottom
        *bottom = *currentStates[0];
        assert(bottom->selectedValues.isFull());
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
            sizeof(uint32_t) * width * fanout + // Indices
            8 * 6; // Alignment
    }

    template<typename ModelType, typename ProblemType, typename StateType>
    void MDD<ModelType, ProblemType, StateType>::printStates(LightVector<StateType> const * states) const
    {
        for(StateType* state = states->begin(); state < states->end(); state += 1)
        {
            state->print();
        }
    }
}