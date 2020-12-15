#pragma once

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

#include "../DP/VRPState.cuh"
#include "../OP/VRP.cuh"

namespace MDD
{
    enum Type {Relaxed, Restricted};

    template<typename ProblemType>
    class MetaMDD
    {
        public:
            unsigned int width;
            unsigned int fanout;
            ProblemType const & problem;

        public:
            MetaMDD(ProblemType const & problem, unsigned int width);

            __host__ __device__ void buildTopDown(OP::VRP const* problem, unsigned int maxWidth, MDDType type, DP::VRPState& top, unsigned int cutsetMaxSize, unsigned int& cutsetSize, DP::VRPState* cutset, DP::VRPState& bottom, std::byte* scratchpadMem);
            static unsigned int calcFanout(ProblemType const & problem);
    };

    template<typename T>
    MetaMDD<T>::MetaMDD(T const & problem, unsigned int width) :
        width(width),
        fanout(calcFanout(problem)),
        problem(problem)
    {}



    __host__ __device__
    void MDD::MetaMDD::buildTopDown(MDD::MDDType type, DP::VRPState& top, unsigned int& cutsetSize, DP::VRPState* cutsetBuffer, DP::VRPState& bottom, std::byte* scratchpadMem)
    {
        //Current states buffer
        RuntimeArray<DP::VRPState> currentStatesBuffer(maxWidth, scratchpadMem);
        unsigned int stateStorageSize = DP::VRPState::sizeOfStorage(problem);
        RuntimeArray<std::byte> currentStatesStorages(stateStorageSize * maxWidth,  Memory::align(4, currentStatesBuffer.storageEnd()));
        for(unsigned int currentStateIdx = 0; currentStateIdx < currentStatesBuffer.getCapacity(); currentStateIdx += 1)
        {
            new (&currentStatesBuffer[currentStateIdx]) DP::VRPState(problem, &currentStatesStorages[stateStorageSize * currentStateIdx]);
        };

        //Next states buffer
        RuntimeArray<DP::VRPState> nextStatesBuffer(maxWidth, Memory::align(4, currentStatesStorages.storageEnd()));
        RuntimeArray<std::byte> nextStatesStorage(stateStorageSize * maxWidth, Memory::align(4, nextStatesBuffer.storageEnd()));
        for(unsigned int nextStateIdx = 0; nextStateIdx < nextStatesBuffer.getCapacity(); nextStateIdx += 1)
        {
            new (&nextStatesBuffer[nextStateIdx]) DP::VRPState(problem, &nextStatesStorage[stateStorageSize * nextStateIdx]);
        };

        //Auxiliary information
        unsigned int fanout = cutsetMaxSize / maxWidth;
        RuntimeArray<uint32_t> costs(fanout * maxWidth, Memory::align(4, nextStatesStorage.storageEnd()));
        RuntimeArray<uint8_t> indices(fanout * maxWidth, Memory::align(4, costs.storageEnd()));

        assert(indices.storageEnd() < scratchpadMem + MDD_SCRATCHPAD_SIZE);

        //Root
        currentStatesBuffer[0] = top;

        //Build
        bool cutsetInitialized = false;
        unsigned int currentStatesCount = 1;
        unsigned int nextStatesCount = 0;
        for(unsigned int varIndex = top.selectedValues.getSize(); varIndex < problem->variables.getCapacity(); varIndex += 1)
        {
            //Initialize indices
            thrust::sequence(thrust::seq, indices.begin(), indices.end());

            //Initialize costs
            thrust::fill(costs.begin(), costs.end(), UINT32_MAX);

            //Calculate costs
            assert(currentStatesCount <= currentStatesBuffer.getCapacity());
            for(unsigned int currentStateIdx = 0; currentStateIdx < currentStatesCount; currentStateIdx += 1)
            {
                DP::VRPModel::calcCosts(problem, varIndex, &currentStatesBuffer[currentStateIdx], &costs[fanout * currentStateIdx]);
            }

            //Sort indices by costs
            thrust::sort_by_key(thrust::seq, costs.begin(), costs.end(), indices.begin());

            //Count next states
            auto* costsEnd = thrust::lower_bound(thrust::seq, costs.begin(), costs.end(), UINT32_MAX);
            unsigned int costsCount = thrust::distance(costs.begin(), costsEnd);

            nextStatesCount = min(maxWidth, costsCount);
            if(varIndex == problem->variables.getCapacity() - 1)
            {
                nextStatesCount = 1;
            }

            //Cutset
            if(costsCount > maxWidth and type == MDDType::Relaxed and (not cutsetInitialized))
            {
                thrust::for_each(thrust::seq, indices.begin(), indices.begin() + costsCount, [=] (uint8_t& index)
                {
                    unsigned int currentStateIdx = index / fanout;
                    unsigned int edgeIdx = index % fanout;
                    int value = problem->variables[varIndex].minValue + edgeIdx;
                    unsigned int cutsetStateIdx = thrust::distance(indices.begin(), &index);
                    DP::VRPModel::makeState(problem,
                                            &currentStatesBuffer[currentStateIdx],
                                            value,
                                            costs[cutsetStateIdx],
                                            &cutset[cutsetStateIdx]);
                });

                cutsetSize = costsCount;
                cutsetInitialized = true;
            }

            //Add states
            assert(nextStatesCount <= indices.getCapacity());
            thrust::for_each(thrust::seq, indices.begin(), indices.begin() + costsCount, [=] (uint8_t& index)
            {
                unsigned int currentStateIdx = index / fanout;
                unsigned int edgeIdx =  index % fanout;
                int value = problem->variables[varIndex].minValue + edgeIdx;
                unsigned int nextStateIdx = thrust::distance(indices.begin(), &index);
                if(nextStateIdx < nextStatesCount)
                {
                    DP::VRPModel::makeState(problem,
                                            &currentStatesBuffer[currentStateIdx],
                                            value,
                                            costs[nextStateIdx],
                                            &nextStatesBuffer[nextStateIdx]);
                    if(type == Restricted)
                    {
                        assert(currentStatesBuffer[currentStateIdx].selectedValues.getSize() == nextStatesBuffer[nextStateIdx].selectedValues.getSize() - 1);
                    }
                }
                else if (type == Relaxed)
                {
                    DP::VRPModel::mergeNextState(problem, &currentStatesBuffer[currentStateIdx], value, &nextStatesBuffer[nextStatesCount - 1]);
                }
            });

            //Prepare for the next loop
            currentStatesBuffer.swap(nextStatesBuffer);
            currentStatesCount = nextStatesCount;
            nextStatesCount = 0;
        }

        //Copy bottom
        bottom = currentStatesBuffer[0];
        if(type == Restricted)
        {
            assert(bottom.selectedValues.isFull());
        }
    }

    template<typename T>
    unsigned int MetaMDD::calcFanout(T const & problem)
    {
        return thrust::transform_reduce(thrust::seq, problem.variables.begin(), problem.variables.end(), OP::Variable::cardinality, 0, thrust::maximum<unsigned int>());
    }
}