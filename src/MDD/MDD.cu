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
#include "Edge.cuh"

using namespace DP;

__device__
MDD::MDD::MDD(Type type, unsigned int width, DP::TSPState const * top, OP::TSPProblem const * problem) :
    type(type),
    width(width),
    fanout(calcFanout(problem)),
    top(top),
    problem(problem)
{}

__host__ __device__
unsigned int MDD::MDD::calcFanout(OP::TSPProblem const * problem)
{
    unsigned int fanout = 0;

    for(OP::Variable * var = problem->vars.begin(); var != problem->vars.end(); var += 1)
    {
        fanout = max(OP::Variable::cardinality(*var), fanout);
    }
    return fanout;
}

__device__
void MDD::MDD::buildTopDown(DP::TSPState* bottom, unsigned int& cutsetSize, DP::TSPState * const cutset, std::byte* buffer)
{
    //Current states buffer
    unsigned int stateStorageSize = DP::TSPState::sizeofStorage(problem);
    RuntimeArray<DP::TSPState> states(width, buffer);
    RuntimeArray<std::byte> statesStorage(stateStorageSize * width, states.getStorageEnd(4));
    thrust::for_each(thrust::seq, states.begin(), states.end(), [=] (auto& state)
    {
        unsigned int stateIdx = thrust::distance(states.begin(), &state);
        new (&state) DP::TSPState(problem, &statesStorage[stateStorageSize * stateIdx]);
    });

    //Next states buffer
    RuntimeArray<DP::TSPState> nextStates(width, statesStorage.getStorageEnd(4));
    RuntimeArray<std::byte> nextStatesStorage(stateStorageSize * width , nextStates.getStorageEnd(4));
    thrust::for_each(thrust::seq, nextStates.begin(), nextStates.end(), [=] (auto& nextState)
    {
        unsigned int nextStateIdx = thrust::distance(nextStates.begin(), &nextState);
        new (&nextState) DP::TSPState(problem , &nextStatesStorage[stateStorageSize * nextStateIdx]);
    });

    //Edges buffer
    RuntimeArray<Edge> edges(width * fanout, nextStatesStorage.getStorageEnd(4));
    thrust::for_each(thrust::seq, edges.begin(), edges.end(), [=] (auto& edge)
    {
        new (&edge) Edge();
    });

    //Auxiliary information
    unsigned int infoCount = fanout * width;
    RuntimeArray<uint32_t> costs(infoCount, edges.getStorageEnd(4));
    RuntimeArray<uint32_t> indices(infoCount, costs.getStorageEnd(4));

    //Root
    states[0] = *top;

    //Build
    bool cutsetInitialized = false;
    unsigned int statesCount = 1;
    unsigned int nextStatesCount = 0;
    for(unsigned int level = top->selectedValues.getSize(); level < problem->vars.size; level += 1 )
    {
        //Initialize indices
        thrust::sequence(thrust::seq, indices.begin(), indices.end());

        //Initialize costs
        thrust::uninitialized_fill(thrust::seq, costs.begin(), costs.end(), UINT32_MAX);

        //Calculate costs
        assert(statesCount <= states.size);
        thrust::for_each(thrust::seq, states.begin(), states.begin() + statesCount, [=] (auto& currentState)
        {
            unsigned int currentStateIdx = thrust::distance(states.begin(), &currentState);
            DP::TSPModel::calcCosts(problem, level, &currentState, &costs[fanout * currentStateIdx]);
        });

        //Sort indices by costs
        thrust::sort_by_key(thrust::seq, costs.begin(), costs.end(), indices.begin());

        //Count next states
        unsigned int* costsEnd = thrust::lower_bound(thrust::seq, costs.begin(), costs.end(), UINT32_MAX);
        unsigned int costsCount = thrust::distance(costs.begin(), costsEnd);

        nextStatesCount = min(width, costsCount);
        nextStatesCount = level < problem->vars.size - 1 ? nextStatesCount : 1;

        //Cutset
        if(costsCount > width and type == MDD::Relaxed and (not cutsetInitialized))
        {
            thrust::for_each(thrust::seq, indices.begin(), indices.begin() + costsCount, [=] (auto& index)
            {
                unsigned int currentStateIdx = index / fanout;
                unsigned int edgeIdx =  index % fanout;
                int value = problem->vars[level].minValue + edgeIdx;
                unsigned int nextStateIdx = thrust::distance(indices.begin(), &index);
                DP::TSPModel::makeNextState(problem, &states[currentStateIdx], value, costs[nextStateIdx], &cutset[nextStateIdx]);
            });

            cutsetSize = costsCount;
            cutsetInitialized = true;
        }

        //Add states
        assert(nextStatesCount <= indices.size);
        thrust::for_each(thrust::seq, indices.begin(), indices.begin() + costsCount, [=] (auto& index)
        {
            unsigned int currentStateIdx = index / fanout;
            unsigned int edgeIdx =  index % fanout;
            int value = problem->vars[level].minValue + edgeIdx;
            unsigned int nextStateIdx = thrust::distance(indices.begin(), &index);
            if(nextStateIdx < nextStatesCount)
            {
                DP::TSPModel::makeNextState(problem, &states[currentStateIdx], value, costs[nextStateIdx], &nextStates[nextStateIdx]);
            }
            else if (type == Relaxed)
            {
                DP::TSPModel::mergeNextState(problem, &states[currentStateIdx], value, &nextStates[nextStatesCount - 1]);
            }
        });

        //Add edges
        assert(costsCount <= indices.size);
        thrust::for_each(thrust::seq, indices.begin(), indices.begin() + costsCount, [=] (auto& index)
        {
            unsigned int nextStateIdx = thrust::distance(indices.begin(), &index);
            if(nextStateIdx < nextStatesCount)
            {
                new(&edges[index]) Edge(nextStateIdx);
            }
            else if (type == Relaxed)
            {
                new (&edges[index]) Edge(nextStatesCount - 1);
            }
        });

        //Prepare for the next loop
        states.swap(nextStates);
        statesCount = nextStatesCount;
        nextStatesCount = 0;
        thrust::for_each(thrust::seq, nextStates.begin(), nextStates.end(), [=] (auto& state)
        {
           DP::TSPState::reset(state);
        });
        thrust::for_each(thrust::seq, edges.begin(), edges.end(), [=] (auto& edge)
        {
           Edge::reset(edge);
        });
    }

    //Copy bottom
    *bottom = states[0];
}