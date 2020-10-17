#include "mdd_optimization/all_different.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>  // for std::invalid_argument
#include <utility>    // for std::move

// #define DEBUG

namespace mdd {

AllDifferentState::AllDifferentState()
: DPState()
{
}

AllDifferentState::AllDifferentState(const AllDifferentState& other)
{
  pStatesList = other.pStatesList;
}

AllDifferentState::AllDifferentState(AllDifferentState&& other)
{
  pStatesList = std::move(other.pStatesList);
}

AllDifferentState& AllDifferentState::operator=(const AllDifferentState& other)
{
  if (&other == this)
  {
    return *this;
  }

  pStatesList = other.pStatesList;
  return *this;
}

AllDifferentState& AllDifferentState::operator=(AllDifferentState&& other)
{
  if (&other == this)
  {
    return *this;
  }

  pStatesList = std::move(other.pStatesList);
  return *this;
}

bool AllDifferentState::isEqual(const DPState* other) const noexcept
{
  auto otherDPState = reinterpret_cast<const AllDifferentState*>(other);

  // Check if "other" is contained in this states
  if (pStatesList.size() < otherDPState->pStatesList.size())
  {
    // Return, there is at least one state in "other" that this DP doesn't have
    return false;
  }

  // Check that all states in "other" are contained in this state
  const auto& otherList = otherDPState->pStatesList;
  for (const auto& otherSubList : otherList)
  {
    // Check if this subSet is contained in the other state subset list
    if (std::find(pStatesList.begin(), pStatesList.end(), otherSubList) == pStatesList.end())
    {
      // State not found
      return false;
    }
  }

  // All states are present
  return true;
}

bool AllDifferentState::isInfeasible() const noexcept
{
  return pStatesList.empty();
}

DPState::SPtr AllDifferentState::next(int64_t domainElement, DPState*) const noexcept
{
  auto state = std::make_shared<AllDifferentState>();
  if (pStatesList.empty())
  {
    state->pStatesList.resize(1);
    state->pStatesList.back().insert(domainElement);
  }
  else
  {
    for (const auto& subSet : pStatesList)
    {
      // Add the new element to all the subset compatible with it
      if (std::find(subSet.begin(), subSet.end(), domainElement) == subSet.end())
      {
        state->pStatesList.push_back(subSet);
        state->pStatesList.back().insert(domainElement);
      }
    }
  }
  return state;
}

double AllDifferentState::cost(int64_t domainElement, DPState*) const noexcept
{
  return static_cast<double>(domainElement);
}

void AllDifferentState::mergeState(DPState* other) noexcept
{
  if (other == nullptr)
  {
    return;
  }

  auto otherDP = reinterpret_cast<const AllDifferentState*>(other);
  for (const auto& otherSubList : otherDP->pStatesList)
  {
    // Check if the other sublist is already present in the current list
    // and, if not, add it
    if (std::find(pStatesList.begin(), pStatesList.end(), otherSubList) == pStatesList.end())
    {
      pStatesList.push_back(otherSubList);
    }
  }
}

std::string AllDifferentState::toString() const noexcept
{
  std::string out{"{"};
  if (pStatesList.empty())
  {
    out += "}";
    return out;
  }

  for (const auto& sublist : pStatesList)
  {
    out += "{";
    for (auto val : sublist)
    {
      out += std::to_string(val) + ", ";
    }
    out.pop_back();
    out.pop_back();
    out += "}, ";
  }
  out.pop_back();
  out.pop_back();
  out += "}";
  return out;
}

AllDifferent::AllDifferent(const std::string& name)
: MDDConstraint(mdd::ConstraintType::kAllDifferent, name),
  pInitialDPState(std::make_shared<AllDifferentState>())
{
}

std::vector<Node*> AllDifferent::mergeNodeSelect(
        int layer,
        const std::vector<std::vector<Node*>>& mddRepresentation) const noexcept
{
  // For the all different, doesn't change much what nodes to select for merging
  std::vector<Node*> nodesToMerge;
  const auto& nodesLayer = mddRepresentation[layer];
  if (nodesLayer.size() < 2)
  {
    return nodesToMerge;
  }
  nodesToMerge.push_back(nodesLayer[0]);
  nodesToMerge.push_back(nodesLayer[1]);

  return nodesToMerge;
}

Node* AllDifferent::mergeNodes(const std::vector<Node*>& nodesList, Arena* arena) const noexcept
{
  assert(!nodesList.empty());
  assert(arena != nullptr);

  // For all different, merging nodes selected with the "mergeNodeSelect" means merging
  // DP states on exclusive sets of values (e.g., merging {1, 2} and {1, 3})
  // Pick one at random and set it as the DP state of the new node
  auto mergedNode = arena->buildNode(nodesList.at(0)->getLayer(), nodesList.at(0)->getVariable());
  mergedNode->resetDPState(getInitialDPState());

  for (auto node : nodesList)
  {
    // Merge all nodes DP states
    mergedNode->getDPState()->mergeState(node->getDPState());
  }
  return mergedNode;
}

DPState::SPtr AllDifferent::getInitialDPState() const noexcept
{
  return pInitialDPState;
}

NodeDomain AllDifferent::getConstraintValuesForPath(const std::vector<Edge*>& path) const
{
    // std::vector<int> usedValues;
    NodeDomain usedValues;
    for (auto edgeInPath : path)
    {
      // For each edge in the path
      for (auto var : getScope())
      {
        // Check if the variables in the scope of this constraints
        // are at the tail of the current edge
        if (var->getId() == edgeInPath->getTail()->getVariable()->getId() )
        {
           usedValues.addValue( edgeInPath->getValue() );
          //  usedValues.push_back( edgeInPath->getValue() );
           break;
        }
      }
    }

    return usedValues;
}

void AllDifferent::enforceConstraint(Arena* arena, std::vector<std::vector<Node*>>& mddRepresentation,
                                     std::vector<Node*>& newNodesList) const
{

    std::vector<int> conflictingValues;
    std::vector<int> seenValues;
    std::vector<int> constraintIds;


    int firstLayerInConstraint = mddRepresentation.size(); int lastLayerInConstraint = 0;

    // Find which are the values that create conflict in this constraint
    // Find which layers are affected by this all diff constraint
    for (auto var : getScope()) {
        constraintIds.push_back( var->getId() );
        if (var->getId() > lastLayerInConstraint) {
          lastLayerInConstraint = var->getId();
        }
        if (var->getId() < firstLayerInConstraint) {
          firstLayerInConstraint = var->getId();
        }

        // for (int val : var->getAvailableValues()) {
        //     if ( std::count(seenValues.begin(), seenValues.end(), val) ) {
        //         if (std::count(conflictingValues.begin(), conflictingValues.end(), val) == 0) {
        //            conflictingValues.push_back( val );
        //         }
        //     } else {
        //       seenValues.push_back( val );
        //     }
        // }
    }


    for (int layer = firstLayerInConstraint; layer <= lastLayerInConstraint; layer++) {

        std::vector<Node*> nodesInLayer = mddRepresentation.at(layer);
        for ( int nodeIdx = 0; nodeIdx < nodesInLayer.size(); nodeIdx++) {

            auto node = nodesInLayer[nodeIdx];
            const auto& nodeInPaths = node->getIncomingPaths();

            // If the layer is in constraint, remove invalid values for the nodes in the layer
            if (node->getInEdges().size() > 0 && std::count(constraintIds.begin(), constraintIds.end(), layer)) {
                auto inEdge = node->getInEdges()[0];

                // All path to same node should lead to the same count, so use the first one
                const auto& path = nodeInPaths.at(inEdge->getUniqueId()).at(0);
                // std::vector<int> usedValues = getConstraintValuesForPath(path);
                NodeDomain usedValues = getConstraintValuesForPath(path);
                std::vector<Edge*> edgesToRemove;

                for (auto edge : node->getOutEdges()) {
                    if (usedValues.isValueInDomain( edge->getValue() )) {
                        edgesToRemove.push_back( edge );
                    }
                }

                for (auto edge : edgesToRemove) {
                    auto head = edge->getHead();
                    auto tail = edge->getTail();
                    edge->removeEdgeFromNodes();
                    arena->deleteEdge(edge->getUniqueId());

                    // if head does not have incoming edges, it's unreachable so clean
                    if (head->getInEdges().size() == 0 ) {
                      eraseUnfeasibleSuccessors(head, arena, mddRepresentation);
                    }
                }


                // If no outgoing edges, node does not lead to solution so remove incoming edges.
                if (node->getOutEdges().size() == 0) {
                    std::vector<Edge*> inEdges = node->getInEdges();
                    for (auto inEdge : inEdges) {
                        auto parent = inEdge->getTail();
                        inEdge->removeEdgeFromNodes();
                        arena->deleteEdge(inEdge->getUniqueId());

                        // If after removing incoming parent does not have outgoing, clean up predecessors.
                        if (parent->getOutEdges().size() == 0) {
                            eraseUnfeasiblePredecessors(parent, arena, mddRepresentation);
                        }
                    }

                    // Finally, remove node because it does not lead to a solution.
                    mddRepresentation[node->getLayer()].erase( std::find(mddRepresentation[node->getLayer()].begin(), 
                        mddRepresentation[node->getLayer()].end(), node) );
                    arena->deleteNode( node->getUniqueId() ); 
                }
            }
        }
            // ************************************************ //

            // If current layer is part of constraint, at this point only valid edges remain.
            // Now, if previous layer was part of constraint, split each node per valid value.

        if ( layer+1 == mddRepresentation.size()-1  ) {
            continue;
        }

        nodesInLayer = mddRepresentation.at(layer+1);
        for ( int nodeIdx = 0; nodeIdx < nodesInLayer.size(); nodeIdx++) {

            auto node = nodesInLayer[nodeIdx];
            const auto& nodeInPaths = node->getIncomingPaths();
            // First split and merge nodes according to their paths (state of constraint)
            spp::sparse_hash_map< int, Node*> nodeByUsedValuesCount;
            std::vector< NodeDomain > usedValuesVector;

            std::vector<Edge*> inEdges = node->getInEdges();
            for (int edgeIdx = 0; edgeIdx < inEdges.size(); edgeIdx++) {
              auto inEdge = inEdges[edgeIdx];

              // All path to same node should lead to the same count, so use the first one
              const auto& path = nodeInPaths.at(inEdge->getUniqueId()).at(0);
              NodeDomain usedValues = getConstraintValuesForPath(path);

              int usedValueId = -1;
              for (int k = 0; k < usedValuesVector.size(); k++ ) {
                  if ( usedValues == usedValuesVector[k] ) {
                    usedValueId = k;
                    break;
                  }
              }

              // if (nodeByUsedValuesCount.find( usedValues ) != nodeByUsedValuesCount.end() ) {
              if ( usedValueId > -1 ) {

                // Equivalent node exist
                inEdge->setHead(nodeByUsedValuesCount[usedValueId]);
              } else {
                  if (edgeIdx == 0) {
                      nodeByUsedValuesCount[ usedValuesVector.size() ] = node;
                      usedValuesVector.push_back( usedValues );
                      continue;
                  }

                  // Create new node for incoming edge
                  auto newNode = arena->buildNode(node->getLayer(), node->getVariable());
                  // newNode->initializeNodeDomain();
                  nodeByUsedValuesCount[ usedValuesVector.size() ] = newNode;
                  usedValuesVector.push_back( usedValues );
                  inEdge->setHead(newNode);

                  // Copy outgoing edges for new node
                  for (auto outEdge : node->getOutEdges())
                  {
                    arena->buildEdge(newNode,
                                    outEdge->getHead(),
                                    outEdge->getValue(),
                                    outEdge->getValue());
                  }
                  
                  // Add new node to the mdd representation
                  mddRepresentation[newNode->getLayer()].push_back(newNode);

              }

            }
        }
    }
}



};
