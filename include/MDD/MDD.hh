#pragma once

#include <vector>
#include <string>

#include <CustomTemplateLibrary/CTL.hh>
#include <DP/State.hh>
#include <MDD/Layer.hh>
#include <Problem/Variable.hh>

class MDD
{
    private:
        uint width;
        ctl::RuntimeArray<Layer> layers;

    public:
        MDD(uint width, std::vector<Variable> const &vars);

        void initialize();

        uint getLayersCount() const;

        void toGraphViz(std::string const & nameFileGv) const;

        void DFS(uint indexLayer, uint indexNode, ctl::StaticVector<int> & labels, bool print) const;

        template<typename C>
        void separate();
};


template<typename C>
void MDD::separate()
{
    using State = typename C::State;

    size_t sizeStorageState = State::getSizeStorage(this);
    size_t sizeStoragesStates = width * sizeStorageState;

    auto * storagesStates = static_cast<std::byte*>(malloc(sizeStoragesStates));
    auto * storagesNextStates = static_cast<std::byte*>(malloc(sizeStoragesStates));
    auto * storageWorkingState = static_cast<std::byte*>(malloc(sizeStoragesStates));

    auto * states = new ctl::StaticVector<State>(width);
    auto * nextStates = new ctl::StaticVector<State>(width);
    State * const workingState = new State(DP::State::Uninitialized, sizeStorageState, storageWorkingState);

    // Initialize root
    states->emplaceBack(State::Type::Root, sizeStorageState, storagesStates);

    for(uint indexLayer = 0; indexLayer < layers.size - 1; indexLayer += 1 )
    {
        Layer & layer = layers[indexLayer];
        Layer & nextLayer = layers[indexLayer + 1];

        // Initialize next states
        for (uint i = 0; i < nextLayer.nodes.getSize(); i += 1)
        {
            nextStates->emplaceBack(State::Type::Uninitialized, sizeStorageState, storagesNextStates + sizeStorageState * i);
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
                        auto * eqState = std::find(nextStates->begin(), nextStates->end(), *workingState);
                        if(eqState != nextStates->end())
                        {
                            // Redirect edge
                            uint indexEqState = std::distance(nextStates->begin(), eqState);
                            edge.to = indexEqState;
                        }
                        else
                        {
                            // Create new state
                            nextStates->emplaceBack(workingState->type, sizeStorageState, storagesNextStates + sizeStorageState * nextStates->getSize());
                            nextStates->back() = *workingState;

                            // Create new node
                            nextLayer.nodes.emplaceBack();
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
            Edge * end = std::remove_if(nodeEdges.begin(), nodeEdges.end(), Edge::isNotValid);
            uint size = std::distance(nodeEdges.begin(), end);
            nodeEdges.resize(size);
        }

        std::swap(states, nextStates);
        std::swap(storagesStates, storagesNextStates);
        nextStates->clear();
    }

    free(states);
    free(nextStates);
    free(workingState);

    free(storagesStates);
    free(storagesNextStates);
    free(storageWorkingState);
}

