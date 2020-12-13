#include <cstdint>
#include <cmath>
#include <numeric>
#include <algorithm>

#include <thrust/transform_reduce.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/uninitialized_fill.h>

#include <Containers/RuntimeArray.cuh>
#include <Utils/Memory.cuh>

#include "MDD.cuh"

__host__ __device__
unsigned int MDD::calcFanout(OP::TSPProblem const * problem)
{
    unsigned int fanout = 0;

    for(OP::Variable * var = problem->vars.begin(); var != problem->vars.end(); var += 1)
    {
        fanout = max(OP::Variable::cardinality(*var), fanout);
    }
    return fanout;
}

__device__
void MDD::buildMddTopDown(OP::TSPProblem const* problem, unsigned int maxWidth, MDDType type, DP::TSPState& top, unsigned int cutsetMaxSize, unsigned int& cutsetSize, DP::TSPState* cutset, DP::TSPState& bottom, std::byte* scratchpad)
{
    //Current states buffer
    RuntimeArray<DP::TSPState> currentStatesBuffer(maxWidth, scratchpad);
    unsigned int stateStorageSize = DP::TSPState::sizeOfStorage(problem);
    RuntimeArray<std::byte> currentStatesStorages(stateStorageSize * maxWidth,  Memory::align(4, currentStatesBuffer.storageEnd()));
    for(unsigned int currentStateIdx = 0; currentStateIdx < currentStatesBuffer.getCapacity(); currentStateIdx += 1)
    {
        new (&currentStatesBuffer[currentStateIdx]) DP::TSPState(problem, &currentStatesStorages[stateStorageSize * currentStateIdx]);
    };

    //Next states buffer
    RuntimeArray<DP::TSPState> nextStatesBuffer(maxWidth, Memory::align(4, currentStatesStorages.storageEnd()));
    RuntimeArray<std::byte> nextStatesStorage(stateStorageSize * maxWidth, Memory::align(4, nextStatesBuffer.storageEnd()));
    for(unsigned int nextStateIdx = 0; nextStateIdx < nextStatesBuffer.getCapacity(); nextStateIdx += 1)
    {
        new (&nextStatesBuffer[nextStateIdx]) DP::TSPState(problem, &nextStatesStorage[stateStorageSize * nextStateIdx]);
    };

    //Auxiliary information
    unsigned int fanout = cutsetMaxSize / maxWidth;
    RuntimeArray<int32_t> costs(fanout * maxWidth, Memory::align(4, nextStatesStorage.storageEnd()));
    RuntimeArray<uint32_t> indices(fanout * maxWidth, Memory::align(4, costs.storageEnd()));

    //Root
    currentStatesBuffer[0] = top;

    //Build
    bool cutsetInitialized = false;
    unsigned int currentStatesCount = 1;
    unsigned int nextStatesCount = 0;
    for(unsigned int level = top.selectedValues.getSize(); level < problem->vars.getCapacity(); level += 1)
    {
        //Initialize indices
        for(unsigned int i = 0; i < indices.getCapacity(); i += 1)
        {
            indices.at(i) = i;
        }

        //Initialize costs
        for(unsigned int i = 0; i < costs.getCapacity(); i +=1)
        {
            costs.at(i) = INT32_MAX;
        }

        //Calculate costs
        assert(currentStatesCount <= currentStatesBuffer.getCapacity());
        for(unsigned int currentStateIdx = 0; currentStateIdx < currentStatesCount; currentStateIdx += 1)
        {
            DP::TSPModel::calcCosts(problem, level, &currentStatesBuffer[currentStateIdx], &costs[fanout * currentStateIdx]);
        }

        //Sort indices by costs
        thrust::sort_by_key(thrust::seq, costs.begin(), costs.end(), indices.begin());

        //Count next states
        int* costsEnd = thrust::lower_bound(thrust::seq, costs.begin(), costs.end(), INT32_MAX);
        unsigned int costsCount = thrust::distance(costs.begin(), costsEnd);

        nextStatesCount = min(maxWidth, costsCount);
        nextStatesCount = level < problem->vars.getCapacity() - 1 ? nextStatesCount : 1;

        //Cutset
        if(costsCount > maxWidth and type == MDDType::Relaxed and (not cutsetInitialized))
        {
            thrust::for_each(thrust::seq, indices.begin(), indices.begin() + costsCount, [=] (unsigned int& index)
            {
                unsigned int currentStateIdx = index / fanout;
                unsigned int edgeIdx = index % fanout;
                int value = problem->vars[level].minValue + edgeIdx;
                unsigned int cutsetStateIdx = thrust::distance(indices.begin(), &index);
                DP::TSPModel::makeNextState(problem, &currentStatesBuffer[currentStateIdx], value, costs[cutsetStateIdx], &cutset[cutsetStateIdx]);
            });

            cutsetSize = costsCount;
            cutsetInitialized = true;
        }

        //Add states
        assert(nextStatesCount <= indices.getCapacity());
        thrust::for_each(thrust::seq, indices.begin(), indices.begin() + costsCount, [=] (unsigned int& index)
        {
            unsigned int currentStateIdx = index / fanout;
            unsigned int edgeIdx =  index % fanout;
            int value = problem->vars[level].minValue + edgeIdx;
            unsigned int nextStateIdx = thrust::distance(indices.begin(), &index);
            if(nextStateIdx < nextStatesCount)
            {
                DP::TSPModel::makeNextState(problem, &currentStatesBuffer[currentStateIdx], value, costs[nextStateIdx], &nextStatesBuffer[nextStateIdx]);
            }
            else if (type == Relaxed)
            {
                DP::TSPModel::mergeNextState(problem, &currentStatesBuffer[currentStateIdx], value, &nextStatesBuffer[nextStatesCount - 1]);
            }
        });

        //Prepare for the next loop
        currentStatesBuffer.swap(nextStatesBuffer);
        currentStatesCount = nextStatesCount;
        nextStatesCount = 0;
    }

    //Copy bottom
    bottom = currentStatesBuffer[0];
}