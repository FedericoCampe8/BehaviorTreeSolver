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

        for (int val : var->getAvailableValues()) {
            if ( std::count(seenValues.begin(), seenValues.end(), val) ) {
                if (std::count(conflictingValues.begin(), conflictingValues.end(), val) == 0) {
                   conflictingValues.push_back( val );
                }
            } else {
              seenValues.push_back( val );
            }
        }
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


// void AllDifferent::enforceConstraintForNode(Node* node, Arena* arena,
//                                      std::vector<std::vector<Node*>>& mddRepresentation,
//                                      std::vector<Node*>& newNodesList) const
// {
//   if (node == nullptr)
//   {
//     throw std::invalid_argument("AllDifferent - enforceConstraint: empty pointer to the node");
//   }

//   if (arena == nullptr)
//   {
//     throw std::invalid_argument("AllDifferent - enforceConstraint: empty pointer to the arena");
//   }

//   // Find all children nodes of the current node
//   std::vector<Node*> children;
//   children.reserve(node->getOutEdges().size());

//   // Use a set for quick lookup to avoid duplicate nodes
//   spp::sparse_hash_set<uint32_t> uniqueNodes;
//   for (auto edge : node->getOutEdges())
//   {
//     auto nextNode = edge->getHead();

//     // Check that the node hasn't already been added
//     if ((uniqueNodes.find(nextNode->getUniqueId()) == uniqueNodes.end()) &&
//             (!nextNode->isLeaf()))
//     {
//       uniqueNodes.insert(nextNode->getUniqueId());
//       children.push_back(nextNode);
//     }
//   }

//   // Enforce AllDifferent constraint by splitting nodes
// #ifdef DEBUG
//   std::cout << "check nodes (num) " << children.size() << std::endl;
// #endif

//   for (int nodeIdx{0}; nodeIdx < static_cast<int>(children.size()); ++nodeIdx)
//   {
//     auto nextNode = children.at(nodeIdx);
//     // const auto& availableValuesTail = node->getValues();
//     auto tailDomain = node->getNodeDomain();
//     // auto availableValuesHead = nextNode->getValuesMutable();
//     auto headDomain = nextNode->getNodeDomain();
    
// #ifdef DEBUG
//     std::cout << "Check for " << availableValuesTail.size() << " values\n";
// #endif

//     // Find all conflicting values
//     std::vector<int64_t> conflictingValues;
//     for (auto val : tailDomain->getValues())
//     {
//       if ( headDomain->isValueInDomain(val) )
//       {

// #ifdef DEBUG
//         std::cout << "Value is conflicting " << val << std::endl;
// #endif
//         conflictingValues.push_back(val);
//       }
//     }

// #ifdef DEBUG
//     std::cout << "check edges (num) " << node->getOutEdges().size() << std::endl;
// #endif
  
//     Edge* conflictingEdge = nullptr;
//     for (auto edge : node->getOutEdges())
//     {
//       // If outgoing edge is in conflicting values find its position
//       for (int idx{0}; idx < static_cast<int>(conflictingValues.size()); ++idx)
//       {
//         if (conflictingValues.at(idx) == edge->getValue())
//         {

// #ifdef DEBUG
//           std::cout << "Found a conflicting edge " << edge->getValue() << std::endl;
// #endif
//           conflictingEdge = edge;
//           break;
//         }
//       }

//       if (conflictingEdge != nullptr)
//       {
//         // BUG FIX: the domain of newNode was initialized with the full domain of the variable:
//         //      newNode->initializeNodeDomain();
//         // I believe that it should be initialized with the restricted domain of nextVariable
//         // auto newReducedDomain = *(node->getValuesMutable());
//         auto newReducedDomain = *(node->getNodeDomain());

//         bool result = newReducedDomain.removeValue(conflictingEdge->getValue());
//         assert( result == true );

//         // BUG FIX: before creating a new node, check if the domain/values that would go into
//         // the newNode is already present in another node on the same level.
//         // If so, simply connect the edges
//         Node* mappedNode{nullptr};
//         for (auto newNode : mddRepresentation[nextNode->getLayer()])
//         {
//           // Note: the following works only on ordered list of values.
//           // TODO move from lists to sets/bitsets/ranges
//           if (*newNode->getNodeDomain() == newReducedDomain)
//           {
//             mappedNode = newNode;
//             break;
//           }
//         }

//         if (mappedNode != nullptr)
//         {
//           // A match is found!
//           // Connect the edge to the mapped node and continue
//           // Move incoming conflicting edge from next node to the mapped node
//           nextNode->removeInEdgeGivenPtr(edge);

//           // Set the in edge on the new node.
//           // Note: the head on the edge is set automatically
//           mappedNode->addInEdge(edge);
//         }
//         else
//         {
//           // Edge points to conflicting value, so split node
//           auto newNode = arena->buildNode(nextNode->getLayer(), nextNode->getVariable());

//           // TODO check if it is possible to "build" the domain rather than copying the entire
//           // variable domain and removing values
//           // auto newAvailableValues = newNode->getValuesMutable();
//           // *newAvailableValues = newReducedDomain;

//           // All paths should lead to the same values being used
//           std::vector<int> usedValues;
//           if (node->getInEdges().size() > 0) {
//               Node::IncomingPathList pathList = node->getIncomingPaths();
//               std::vector< Node::EdgeList > paths = pathList[ node->getInEdges()[0]->getUniqueId() ];
//               usedValues = getConstraintValuesForPath( paths[0] );
//           }
//           usedValues.push_back( conflictingEdge->getValue() );

//           for (auto outEdge: nextNode->getOutEdges()) {
//             if (std::count(usedValues.begin(), usedValues.end(), outEdge->getValue()) == 0) {
//                 arena->buildEdge(newNode, outEdge->getHead(), outEdge->getDomainLowerBound(), outEdge->getDomainUpperBound());
//             }
//           }

//           auto newDomain = newNode->getNodeDomain();

//           // If new node has no available values,
//           // then it is infeasible so do not add it to the graph
//           if (newDomain->getSize() > 0)
//           {
//             // Add the node to the current level
//             mddRepresentation[nextNode->getLayer()].push_back(newNode);

//             // Store the new node in the newNodesList for next iteration
//             newNodesList.push_back(newNode);

//             // Move incoming conflicting edge from next node to splitting node
//             nextNode->removeInEdgeGivenPtr(edge);

//             // Set the in edge on the new node.
//             // Note: the head on the edge is set automatically
//             newNode->addInEdge(edge);

//           }
//         }

//         // If nextNode doesn't have any incoming edge, it can be removed
//         // from the layer since it is not reachable anymore
//         if (nextNode->getInEdges().empty())
//         {
//           // Remove all the outgoing edges
//           Node::EdgeList outEdges = nextNode->getOutEdges();
//           for (auto outEdge : outEdges)
//           {
//             outEdge->removeEdgeFromNodes();
//             arena->deleteEdge(outEdge->getUniqueId());
//           }

//           // Remove the node from the layer
//           auto& nextNodeLayer = mddRepresentation[nextNode->getLayer()];
//           auto itNode = std::find(nextNodeLayer.begin(), nextNodeLayer.end(), nextNode);

//           assert(itNode != nextNodeLayer.end());
//           nextNodeLayer.erase(itNode);
//           arena->deleteNode(nextNode->getUniqueId());
//         }
//       }  // position > -1
//     }  // for all out edges
//   }  // for all nodes
// }

};
