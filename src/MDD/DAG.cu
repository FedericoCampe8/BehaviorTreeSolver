#include <cassert>
#include <cstdlib>
#include <new>

#include <thrust/partition.h>
#include <thrust/execution_policy.h>

#include <Utils/Memory.cuh>

#include "DAG.cuh"

__device__
MDD::DAG::DAG( unsigned int width, unsigned int fanout, unsigned int height) :
    width(width),
    fanout(fanout),
    height(height),
    stateStorageSize(DP::TSPState::sizeofStorage(height)),
    edges(width * height * fanout),
    states(width * (height + 1)),
    statesStorage(stateStorageSize * states.size, mallocStatesStorage())
{

    //Initialize edges
    thrust::for_each(thrust::seq, edges.begin(), edges.end(), [=] (auto& edge)
    {
        new (&edge) Edge();
    });

    //Initialize states
    std::size_t stateStorageSize = DP::TSPState::sizeofStorage(height);
    thrust::for_each(thrust::seq, states.begin(), states.end(), [=] (auto& state)
    {
        unsigned int stateIdx = thrust::distance(states.begin(), &state);
        new (&state) DP::TSPState(stateStorageSize, &statesStorage[stateStorageSize * stateIdx]);
    });
}

__device__
std::byte* MDD::DAG::mallocStatesStorage() const
{
    return Memory::safeMalloc(stateStorageSize * states.size);
}

__device__
MDD::Edge* MDD::DAG::getEdges(unsigned int level) const
{
    return &edges[level * width * fanout];
}

__device__
DP::TSPState* MDD::DAG::getStates(unsigned int level) const
{
    unsigned int index = level * width;
    return &states[index];
}

__device__
std::byte* MDD::DAG::getStatesStorage(unsigned int level) const
{
    unsigned int index = level * width * stateStorageSize;
    return &statesStorage[index];
}

__device__
MDD::Edge& MDD::DAG::getEdge(unsigned int level, unsigned int nodeIdx, unsigned int edgeIdx) const
{
    unsigned int index = (level * width * fanout) + (nodeIdx * fanout) + edgeIdx;
    return edges[index];
}

