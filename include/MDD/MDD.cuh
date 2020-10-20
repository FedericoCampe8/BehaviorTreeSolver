#pragma once

#include <cstdint>
#include <utility>

#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include "Extra/Containers/StaticVector.cuh"
#include "Extra/Utils/Memory.cuh"
#include "MDD/Edge.cuh"
#include "Problem/Variable.cuh"

template<typename State>
class MDD
{
    public:
        enum Type {Relaxed, Restricted};
        uint16_t const type;
        uint16_t const height;
        uint16_t const width;
        uint16_t const fanout;
    private:

        std::pair<int16_t,int16_t>* const extremes;
        unsigned int const edgesCount;
        Edge * const edges;
        unsigned int const statesCount;
        std::byte* const statesStorage;
        State* const states;

    public:
        __device__ MDD(Type type, unsigned int width, unsigned int varsCount, Variable const * vars);
        __device__ inline Edge* getEdges(unsigned int layerIdx, unsigned int nodeIdx) const;
        __device__ inline State* getStates(unsigned int layerIdx) const;
        __device__ inline std::pair<int16_t,int16_t> const & getExtremes(unsigned int layerIdx) const;
        __device__ void buildTopDown(std::byte* shrMem);
        __device__ void toGraphViz() const;
        __device__ void DFS(unsigned int  layerIdx, unsigned int nodeIdx, StaticVector<int> & labels, bool print) const;
    private:
        __device__ static uint16_t calculateFanout(unsigned int varsCount, Variable const * vars);
        __device__ static std::pair<int16_t,int16_t>* mallocExtremes(unsigned int height);
        __device__ static Edge* mallocEdges(unsigned int edgesCount);
        __device__ static std::byte* mallocStatesStorage(unsigned int statesCount, unsigned int height);
        __device__ static State* mallocStates(unsigned int statesCount);
};


template<typename State>
__device__
MDD<State>::MDD(Type type, unsigned int width, unsigned int varsCount, Variable const * vars) :
    type(type),
    height(varsCount),
    width(width),
    fanout(calculateFanout(varsCount, vars)),
    extremes(mallocExtremes(height)),
    edgesCount(width * height * fanout),
    edges(mallocEdges(edgesCount)),
    statesCount(width * (height + 1)),
    statesStorage(mallocStatesStorage(height, statesCount)),
    states(mallocStates(statesCount))
{

    // Initialize extremes
    auto initExtremes = [&] (auto & e)
    {
        unsigned int i = thrust::distance(extremes, &e);
        e.first = vars[i].minValue;
        e.second = vars[i].maxValue;
    };
    //Todo: Parallelize
    thrust::for_each(thrust::seq, extremes, &extremes[height], initExtremes);

    // Initialize edges
    auto initEdge = [&] (auto & e)
    {
        new (&e) Edge();
    };
    //Todo: Parallelize
    thrust::for_each(thrust::seq, edges, &edges[edgesCount], initEdge);

    //Initialize states
    std::size_t stateStorageSize = State::sizeofStorage(height);
    auto initStates = [&] (auto & s)
    {
        unsigned int i = thrust::distance(states, &s);
        new (&s) State(stateStorageSize, &statesStorage[stateStorageSize * i]);
    };
    //Todo: Parallelize
    thrust::for_each(thrust::seq, states, &states[statesCount], initStates);
}

template<typename State>
__device__
Edge * MDD<State>::getEdges(unsigned int layerIdx, unsigned int nodeIdx) const
{
    return &edges[layerIdx * width * fanout + nodeIdx * fanout];
}

template<typename State>
__device__
State* MDD<State>::getStates(unsigned int layerIdx) const
{
    return &states[layerIdx * width];
}

template<typename State>
__device__
std::pair<int16_t,int16_t> const & MDD<State>::getExtremes(unsigned int layerIdx) const
{
    return extremes[layerIdx];
}

template<typename State>
__device__ uint16_t MDD<State>::calculateFanout(unsigned int varsCount, Variable const * vars)
{
    //Todo: Parallelize
    unsigned int fanout = thrust::transform_reduce(
        thrust::seq,
        vars,
        &vars[varsCount],
        Variable::cardinality,
        0,
        thrust::maximum<unsigned int>());
    return static_cast<uint16_t>(fanout);
}

template<typename State>
__device__
std::pair<int16_t,int16_t>* MDD<State>::mallocExtremes(unsigned int height)
{
    void * extremes = Memory::alignedMalloc(sizeof(std::pair<int16_t, int16_t>) * height);
    return static_cast<std::pair<int16_t,int16_t>*>(extremes);
}

template<typename State>
__device__
Edge* MDD<State>::mallocEdges(unsigned int edgesCount)
{
    void * edges = malloc(sizeof(Edge) * edgesCount);
    return static_cast<Edge*>(edges);
}

template<typename State>
__device__
std::byte* MDD<State>::mallocStatesStorage(unsigned int height, unsigned int statesCount)
{
    std::size_t stateStorageSize = State::sizeofStorage(height);
    void* statesStorage = Memory::alignedMalloc(stateStorageSize * statesCount);
    return static_cast<std::byte*>(statesStorage);
}

template<typename State>
__device__
State* MDD<State>::mallocStates(unsigned int statesCount)
{
    void * states = Memory::alignedMalloc(sizeof(State) * statesCount);
    return static_cast<State*>(states);
}

template<typename State>
__device__
void MDD<State>::toGraphViz() const
{
    printf("digraph G\n");
    printf("{\n");
    printf("\n");
    printf("  node [shape=circle];\n");

    // Edges
    for (unsigned int layerIdx = 0; layerIdx < height; layerIdx += 1)
    {
        printf("\n");
        for (unsigned int nodeIdx = 0; nodeIdx < width; nodeIdx += 1)
        {
            Edge* edges = getEdges(layerIdx, nodeIdx);
            for (unsigned int edgeIdx = 0; edgeIdx < fanout; edgeIdx += 1)
            {
                Edge const & e = edges[edgeIdx];
                if(e.isActive())
                {
                    printf("  L%dN%d -> L%dN%d  [label=\"%d\"];\n", layerIdx, e.from, layerIdx + 1, e.to, e.value);
                }
            }
        }
    }
    printf("\n");
    printf("}\n");
}

template<typename State>
__device__
void MDD<State>::DFS(unsigned int layerIdx, unsigned int nodeIdx, StaticVector<int> & labels, bool print) const
{
    if(layerIdx == height)
    {
        if (print)
        {
            printf("%d", labels[0]);
            for (uint i = 1; i < layerIdx; i += 1)
            {
                printf(",%d", labels[i]);
            }
            printf("\n");
        }
    }
    else
    {
        Edge * nodeEdges = getEdges(layerIdx, nodeIdx);
        auto & extremes = getExtremes(layerIdx);
        int minValue = extremes.first;
        int maxValue = extremes.second;

        for (int edgeIdx = 0; edgeIdx <= maxValue - minValue; edgeIdx += 1)
        {
            Edge const & e = nodeEdges[edgeIdx];
            if(e.isActive())
            {
                labels.emplaceBack(e.value);
                DFS(layerIdx + 1, e.to, labels, print);
                labels.popBack();
            }
        }
    }
}

template<typename State>
__device__
void MDD<State>::buildTopDown(std::byte* shrMem)
{
    //Temporary states
    std::size_t stateStorageSize = State::sizeofStorage(height);
    std::byte* tmpStatesStorage =  static_cast<std::byte*>(Memory::alignedMalloc(stateStorageSize * fanout));
    State* tmpStates = static_cast<State*>(Memory::alignedMalloc(sizeof(State) * fanout));

    auto initTmpStates = [&] (auto & s)
    {
        unsigned int i = thrust::distance(tmpStates, &s);
        new (&s) State(stateStorageSize, &tmpStatesStorage[stateStorageSize * i]);
    };
    thrust::for_each(thrust::seq, tmpStates, &tmpStates[fanout], initTmpStates);

    //Similarity
    unsigned int* similarityScores = static_cast<unsigned int*>(Memory::alignedMalloc(sizeof(unsigned int) * width));

    //Build
    State::makeRoot(&states[0]);
    unsigned int currentStatesCount = 1;
    unsigned int nextStatesCount = 0;
    for(unsigned int layerIdx = 0; layerIdx < height; layerIdx += 1 )
    {
        State* currentStates = getStates(layerIdx);
        State* nextStates = getStates(layerIdx + 1);

        auto & extremes = getExtremes(layerIdx);
        int minValue = extremes.first;
        int maxValue = extremes.second;

        for(unsigned int currentStateIdx = 0; currentStateIdx < currentStatesCount; currentStateIdx += 1)
        {
            State* currentState = &currentStates[currentStateIdx];
            Edge* currentEdges = getEdges(layerIdx, currentStateIdx);
            State::getNextStates(currentState, minValue, maxValue, tmpStates);

            for (unsigned int edgeIdx = 0; edgeIdx <= maxValue - minValue; edgeIdx += 1)
            {
                int value = minValue + edgeIdx;
                State* tmpState = &tmpStates[edgeIdx];

                if (tmpState->type == State::Type::Regular)
                {

                    State * eqNextState = thrust::find(thrust::seq, nextStates, &nextStates[nextStatesCount], *tmpState);
                    if (eqNextState != &nextStates[nextStatesCount])
                    {
                        unsigned int nextStateIdx = Algorithms::distance(nextStates, eqNextState);
                        new (&currentEdges[edgeIdx]) Edge(currentStateIdx, nextStateIdx, value);
                    }
                    else
                    {
                        if(nextStatesCount < width)
                        {
                            unsigned int nextStateIdx = nextStatesCount;
                            nextStates[nextStateIdx] = *tmpState;
                            nextStatesCount += 1;
                            new (&currentEdges[edgeIdx]) Edge(currentStateIdx, nextStateIdx, value);
                        }
                        else if (type == Type::Relaxed)
                        {
                            //Todo: Parallelize
                            auto getSimilarity = [&] (auto & s) -> unsigned int
                            {
                                return  tmpState->getSimilarity(s);
                            };
                            thrust::transform(thrust::seq, nextStates, &nextStates[nextStatesCount], similarityScores, getSimilarity);
                            unsigned int * mostSimilarScore = thrust::max_element(thrust::seq, similarityScores, &similarityScores[nextStatesCount]);
                            unsigned int nextStateIdx = Algorithms::distance(similarityScores, mostSimilarScore);
                            new (&currentEdges[edgeIdx]) Edge(currentStateIdx, nextStateIdx, value);
                        }
                    }
                }
                else
                {
                    //Deactivate the edge
                    currentEdges[edgeIdx].deactivate();
                }

            }
        }

        currentStatesCount = nextStatesCount;
        nextStatesCount = 0;
    }

    free(reinterpret_cast<void*>(tmpStatesStorage));
    free(reinterpret_cast<void*>(tmpStates));
}