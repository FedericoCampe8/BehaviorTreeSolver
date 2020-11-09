#include <cstdint>
#include <cmath>

#include <thrust/transform_reduce.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/uninitialized_fill.h>

#include <Containers/RuntimeArray.cuh>
#include <Utils/Memory.cuh>

#include "MDD.cuh"

using namespace DP;

__device__
MDD::MDD::MDD(Type type, unsigned int width, DP::TSPState const * const root, unsigned int rootLvl, OP::TSPProblem const * problem) :
    type(type),
    root(root),
    rootLvl(rootLvl),
    problem(problem),
    dag(width, calcFanout(), calcHeight())
{}

__device__
unsigned int MDD::MDD::calcFanout() const
{
    return thrust::transform_reduce(
        thrust::seq,
        &problem->vars[rootLvl],
        problem->vars.end(),
        OP::Variable::cardinality,
        0,
        thrust::maximum<unsigned int>());
}

__device__
unsigned int MDD::MDD::calcHeight() const
{
    return problem->vars.size - rootLvl;
}


__device__
void MDD::MDD::buildTopDown(std::byte* buffer)
{
    //Current states buffer
    RuntimeArray<DP::TSPState> states(dag.width, buffer);
    RuntimeArray<std::byte> statesStorage(dag.width * dag.stateStorageSize, states.getStorageEnd());
    thrust::for_each(thrust::seq, states.begin(), states.end(), [=] (auto& state)
    {
        unsigned int stateIdx = thrust::distance(states.begin(), &state);
        new (&state) DP::TSPState(dag.height, &statesStorage[dag.stateStorageSize * stateIdx]);
    });

    //Next states buffer
    RuntimeArray<DP::TSPState> nextStates(dag.width, statesStorage.getStorageEnd());
    RuntimeArray<std::byte> nextStatesStorage(dag.width * dag.stateStorageSize, nextStates.getStorageEnd());
    thrust::for_each(thrust::seq, nextStates.begin(), nextStates.end(), [=] (auto& state)
    {
        unsigned int stateIdx = thrust::distance(nextStates.begin(), &state);
        new (&state) DP::TSPState(dag.height, &nextStatesStorage[dag.stateStorageSize * stateIdx]);
    });

    //Edges buffer
    RuntimeArray<Edge> edges(dag.width * dag.fanout, nextStatesStorage.getStorageEnd());
    thrust::for_each(thrust::seq, edges.begin(), edges.end(), [=] (auto& edge)
    {
        new (&edge) Edge();
    });

    //Auxiliary information
    unsigned int infoCount = dag.fanout * dag.width;
    RuntimeArray<uint32_t> costs(infoCount, edges.getStorageEnd());
    RuntimeArray<uint32_t> indices(infoCount, costs.getStorageEnd());

    //Root
    states[0] = *root;

    //Copy fist level to global
    thrust::for_each(thrust::seq, states.begin(), states.end(), [=] (auto& state)
    {
        unsigned int stateIdx = thrust::distance(states.begin(), &state);
        dag.getStates(0)[stateIdx] = state;
    });

    //Build
    unsigned int statesCount = 1;
    unsigned int nextStatesCount = 0;
    for(unsigned int level = 0; level < dag.height; level += 1 )
    {
        //Intitialize indices
        thrust::sequence(thrust::seq, indices.begin(), indices.end());

        //Initialize costs
        thrust::uninitialized_fill(thrust::seq, costs.begin(), costs.end(), UINT32_MAX);

        //Calculate costs
        assert(statesCount <= states.size);
        thrust::for_each(thrust::seq, states.begin(), states.begin() + statesCount, [=] (auto& currentState)
        {
            unsigned int currentStateIdx = thrust::distance(states.begin(), &currentState);
            DP::TSPModel::calcCosts(problem, rootLvl + level, &currentState, &costs[currentStateIdx * dag.fanout]);
        });

        //Sort indices by costs
        thrust::sort_by_key(thrust::seq, costs.begin(), costs.end(), indices.begin());

        //Count next states
        unsigned int* costsEnd = thrust::lower_bound(thrust::seq, costs.begin(), costs.end(), UINT32_MAX);
        unsigned int costsCount = thrust::distance(costs.begin(), costsEnd);
        nextStatesCount = min(dag.width, costsCount);
        nextStatesCount = level < dag.height - 1 ? nextStatesCount : 1;

        //Add states
        assert(nextStatesCount <= indices.size);
        thrust::for_each(thrust::seq, indices.begin(), indices.begin() + nextStatesCount, [=] (auto& index)
        {
            unsigned int currentStateIdx = index / dag.fanout;
            unsigned int edgeIdx =  index % dag.fanout;
            int value = problem->vars[rootLvl + level].minValue + edgeIdx;
            unsigned int nextStateIdx = thrust::distance(indices.begin(), &index);
            DP::TSPModel::makeNextState(problem, &states[currentStateIdx], value, costs[nextStateIdx], &nextStates[nextStateIdx]);
        });

        //Add edges
        assert(costsCount <= indices.size);
        thrust::for_each(thrust::seq, indices.begin(), indices.begin() + costsCount, [=] (auto& index)
        {
            unsigned int nextStateIdx = thrust::distance(indices.begin(), &index);
            if(type == Relaxed or nextStateIdx <= dag.width - 1)
            {
                if(level < dag.height - 1)
                {
                    nextStateIdx = min(dag.width - 1, nextStateIdx);
                }
                else
                {
                    nextStateIdx = 0;
                };

                new (&edges[index]) Edge(nextStateIdx);
            }
        });

        //Copy to global
        thrust::for_each(thrust::seq, nextStates.begin(), nextStates.end(), [=] (auto& nextState)
        {
            unsigned int stateIdx = thrust::distance(nextStates.begin(), &nextState);
            dag.getStates(level + 1)[stateIdx] = nextState;
        });
        thrust::for_each(thrust::seq, edges.begin(), edges.end(), [=] (auto& edge)
        {
            unsigned int edgeIdx = thrust::distance(edges.begin(), &edge);
            dag.getEdges(level)[edgeIdx] = edge;
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
}


__device__
void MDD::MDD::print(unsigned int rootLvl, bool endline) const
{
    printf("digraph G\n");
    printf("{\n");
    printf("\n");
    printf("  node [shape=circle];\n");

    //States
    for (unsigned int level = 0; level < dag.height; level += 1)
    {
        printf("\n");
        printf("  {\n");
        printf("      rank = same;\n");

        DP::TSPState* states = dag.getStates(level);
        unsigned int previousStateIdx = 0;
        for(unsigned int stateIdx = 0; stateIdx < dag.width; stateIdx += 1)
        {
            DP::TSPState const & state = states[stateIdx];
            if (DP::TSPState::isActive(state))
            {
                if (stateIdx > 0)
                {
                    printf("      L%dN%d -> L%dN%d [style=invis];\n", level, previousStateIdx, level, stateIdx);
                    previousStateIdx = stateIdx;
                }
            }
        }
        printf("  }\n");
    }

    //Edges
    for (unsigned int level = 0; level < dag.height; level += 1)
    {
        printf("\n");

        DP::TSPState* states = dag.getStates(level);
        for(unsigned int stateIdx = 0; stateIdx < dag.width; stateIdx += 1)
        {
            DP::TSPState const & state = states[stateIdx];
            if (DP::TSPState::isActive(state))
            {
                for (unsigned int edgeIdx = 0; edgeIdx < dag.fanout; edgeIdx += 1)
                {
                    Edge& edge = dag.getEdge(level, stateIdx, edgeIdx);
                    if (Edge::isActive(edge))
                    {
                        printf("  L%dN%d -> L%dN%d [label=\"%d\"];\n", level, stateIdx, level + 1, edge.to, problem->vars[rootLvl + level].minValue + edgeIdx);
                    }
                }
            }
        }
    }

    printf("\n");
    printf("}");

    if (endline)
    {
        printf("\n");
    }
}