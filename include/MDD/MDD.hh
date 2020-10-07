#pragma once

#include <Extra/Extra.hh>
#include <Problem/State.hh>
#include <MDD/Layer.hh>
#include <Problem/Variable.hh>

class MDD
{
    private:
        uint width;
        Extra::Containers::RestrainedArray<Layer> layers;
        uint IDNextNode;

    public:
        __device__ MDD(uint width, uint countVars, Problem::Variable const * const vars);
        __device__ uint getLayersCount() const;
        __device__ void toGraphViz() const;
        __device__ void DFS(uint indexLayer, uint indexNode, Extra::Containers::RestrainedVector<int> & labels, bool print) const;
        template<typename T> __device__ void separation();
        template<typename T> __device__ void topDown();
    private:
        __device__ void initializeSeparation();
};

template<typename T>
__device__
void MDD::separation()
{
    initializeSeparation();

    using State = typename T::State;

    size_t sizeStorageState = State::getSizeStorage(this);
    size_t sizeStorageStates = width * sizeStorageState;

    auto * storageStates = static_cast<std::byte*>(Extra::Utils::Memory::malloc(sizeStorageStates));
    auto * storageNextStates = static_cast<std::byte*>(Extra::Utils::Memory::malloc(sizeStorageStates));
    auto * storageWorkingState = static_cast<std::byte*>(Extra::Utils::Memory::malloc(sizeStorageState));

    auto * states = new Extra::Containers::RestrainedVector<State>(width);
    auto * nextStates = new Extra::Containers::RestrainedVector<State>(width);
    State * const workingState = new State(Problem::State::Uninitialized, sizeStorageState, storageWorkingState);

    // Initialize root
    states->emplaceBack(State::Type::Root, sizeStorageState, storageStates);
    uint indexLastLayer = layers.size - 1;
    for(uint indexLayer = 0; indexLayer < indexLastLayer; indexLayer += 1 )
    {
        Layer & layer = layers[indexLayer];
        Layer & nextLayer = layers[indexLayer + 1];

        // Initialize next states
        for (uint i = 0; i < nextLayer.nodes.getSize(); i += 1)
        {
            nextStates->emplaceBack(State::Type::Uninitialized, sizeStorageState, &storageNextStates[sizeStorageState * i]);
        }

        // Separation
        for(uint indexNode = 0; indexNode < layer.nodes.getSize(); indexNode += 1)
        {
            State const & parentState = states->at(indexNode);
            auto & nodeEdges = layer.edges[indexNode];

            for(uint edgeIndex = 0; edgeIndex < nodeEdges.getSize(); edgeIndex += 1)
            {
                Edge & edge = nodeEdges[edgeIndex];
                State & childState = nextStates->at(edge.to);

                parentState.next(edge.value, workingState);

                if(workingState->type == State::Type::Impossible)
                {
                    // Mark edge as invalid
                    edge.status = Edge::Status::Invalid;
                }
                else if (childState.type == State::Type::Uninitialized)
                {
                    childState = *workingState;
                }
                else if (childState.type == State::Type::Regular)
                {
                    if (not (childState == *workingState))
                    {
                        State * eqState;
#ifdef __CUDA_ARCH__
                        eqState = thrust::find(thrust::device, nextStates->begin(), nextStates->end(), *workingState);
#else
                        eqState = std::find(nextStates->begin(), nextStates->end(), *workingState);
#endif
                        if(eqState != nextStates->end())
                        {
                            // Redirect edge
                            uint indexEqState;
#ifdef __CUDA_ARCH__
                            indexEqState = thrust::distance(nextStates->begin(), eqState);
#else
                            indexEqState = std::distance(nextStates->begin(), eqState);
#endif
                            edge.to = indexEqState;
                        }
                        else
                        {
                            // Create new state
                            nextStates->emplaceBack(workingState->type, sizeStorageState, &storageNextStates[sizeStorageState * nextStates->getSize()]);
                            nextStates->back() = *workingState;

                            // Create new node
                            nextLayer.nodes.emplaceBack(IDNextNode++);
                            uint nodeIndex = nextLayer.nodes.getSize() - 1;

                            //Copy edges
                            nextLayer.edges[nodeIndex] = nextLayer.edges[edge.to];

                            // Redirect edge
                            edge.to = nodeIndex;
                        }
                    }
                }
            }

            //Remove invalid edges
            Edge * end = Extra::Algorithms::remove_if(nodeEdges.begin(), nodeEdges.end(), Edge::isNotValid);
            uint size = Extra::Algorithms::distance(nodeEdges.begin(), end);

            nodeEdges.resize(size);
        }
        Extra::Algorithms::swap(states, nextStates);
        Extra::Algorithms::swap(storageStates, storageNextStates);

        nextStates->clear();
    }

    delete states;
    delete nextStates;
    delete workingState;

    Extra::Utils::Memory::free(storageStates);
    Extra::Utils::Memory::free(storageNextStates);
    Extra::Utils::Memory::free(storageWorkingState);
}


template<typename T>
__device__
void MDD::topDown()
{
    using State = typename T::State;

    size_t sizeStorageState = State::getSizeStorage(this);
    size_t sizeStorageStates = width * sizeStorageState;

    auto * storageStates = static_cast<std::byte*>(Extra::Utils::Memory::malloc(sizeStorageStates));
    auto * storageNextStates = static_cast<std::byte*>(Extra::Utils::Memory::malloc(sizeStorageStates));
    auto * storageWorkingState = static_cast<std::byte*>(Extra::Utils::Memory::malloc(sizeStorageState));

    auto * states = new Extra::Containers::RestrainedVector<State>(width);
    auto * nextStates = new Extra::Containers::RestrainedVector<State>(width);
    State * const workingState = new State(Problem::State::Uninitialized, sizeStorageState, storageWorkingState);

    // Initialize root
    states->emplaceBack(State::Type::Root, sizeStorageState, storageStates);
    layers[0].nodes.emplaceBack(IDNextNode++);

    uint indexLastLayer = layers.size - 1;
    for(uint indexLayer = 0; indexLayer < indexLastLayer; indexLayer += 1 )
    {
        Layer & layer = layers[indexLayer];
        Layer & nextLayer = layers[indexLayer + 1];

        for(uint indexNode = 0; indexNode < layer.nodes.getSize(); indexNode += 1)
        {
            State const &parentState = states->at(indexNode);
            auto &nodeEdges = layer.edges[indexNode];

            for (int value = layer.minValue; value <= layer.maxValue; value += 1)
            {
                parentState.next(value, workingState);
                if (workingState->type == State::Type::Regular)
                {
                    State *eqState;
                    eqState = Extra::Algorithms::find(nextStates->begin(), nextStates->end(), *workingState);
                    if (eqState != nextStates->end())
                    {
                        uint indexEqState = Extra::Algorithms::distance(nextStates->begin(), eqState);
                        nodeEdges.emplaceBack(indexEqState, value);
                    }
                    else
                    {
                        // Create new state
                        nextStates->emplaceBack(workingState->type,
                                                sizeStorageState,
                                                &storageNextStates[sizeStorageState * nextStates->getSize()]);
                        nextStates->back() = *workingState;

                        // Create new node
                        nextLayer.nodes.emplaceBack(IDNextNode++);
                        uint nodeIndex = nextLayer.nodes.getSize() - 1;

                        nodeEdges.emplaceBack(nodeIndex, value);
                    }
                }
            }
        }

        Extra::Algorithms::swap(states, nextStates);
        Extra::Algorithms::swap(storageStates, storageNextStates);
        nextStates->clear();
    }

    delete states;
    delete nextStates;
    delete workingState;

    Extra::Utils::Memory::free(storageStates);
    Extra::Utils::Memory::free(storageNextStates);
    Extra::Utils::Memory::free(storageWorkingState);
}



