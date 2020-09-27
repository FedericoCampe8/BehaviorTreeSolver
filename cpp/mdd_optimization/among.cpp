#include "mdd_optimization/among.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>  // for std::invalid_argument
#include <utility>    // for std::move

// #define DEBUG

namespace {
constexpr int32_t kDefaultBitmapSize{32};
}  // namespace

namespace mdd {

// AmongState::AmongState()
// : DPState()
// {
// }

// AmongState::AmongState(const AmongState& other)
// {
//   pStatesList = other.pStatesList;
// }

// AmongState::AmongState(AmongState&& other)
// {
//   pStatesList = std::move(other.pStatesList);
// }

// AmongState& AmongState::operator=(const AmongState& other)
// {
//   if (&other == this)
//   {
//     return *this;
//   }

//   pStatesList = other.pStatesList;
//   return *this;
// }

// AmongState& AmongState::operator=(AmongState&& other)
// {
//   if (&other == this)
//   {
//     return *this;
//   }

//   pStatesList = std::move(other.pStatesList);
//   return *this;
// }

// bool AmongState::isEqual(const DPState* other) const noexcept
// {
//   auto otherDPState = reinterpret_cast<const AmongState*>(other);

//   // Check if "other" is contained in this states
//   if (pStatesList.size() < otherDPState->pStatesList.size())
//   {
//     // Return, there is at least one state in "other" that this DP doesn't have
//     return false;
//   }

//   // Check that all states in "other" are contained in this state
//   const auto& otherList = otherDPState->pStatesList;
//   for (const auto& otherSubList : otherList)
//   {
//     // Check if this subSet is contained in the other state subset list
//     if (std::find(pStatesList.begin(), pStatesList.end(), otherSubList) == pStatesList.end())
//     {
//       // State not found
//       return false;
//     }
//   }

//   // All states are present
//   return true;
// }

// bool AmongState::isInfeasible() const noexcept
// {
//   return pStatesList.empty();
// }

// DPState::SPtr AmongState::next(int64_t domainElement) const noexcept
// {
//   auto state = std::make_shared<AmongState>();
//   if (pStatesList.empty())
//   {
//     state->pStatesList.resize(1);
//     state->pStatesList.back().insert(domainElement);
//   }
//   else
//   {
//     for (const auto& subSet : pStatesList)
//     {
//       // Add the new element to all the subset compatible with it
//       if (std::find(subSet.begin(), subSet.end(), domainElement) == subSet.end())
//       {
//         state->pStatesList.push_back(subSet);
//         state->pStatesList.back().insert(domainElement);
//       }
//     }
//   }
//   return state;
// }

// double AmongState::cost(int64_t domainElement) const noexcept
// {
//   return static_cast<double>(domainElement);
// }

// void AmongState::mergeState(DPState* other) noexcept
// {
//   if (other == nullptr)
//   {
//     return;
//   }

//   auto otherDP = reinterpret_cast<const AmongState*>(other);
//   for (const auto& otherSubList : otherDP->pStatesList)
//   {
//     // Check if the other sublist is already present in the current list
//     // and, if not, add it
//     if (std::find(pStatesList.begin(), pStatesList.end(), otherSubList) == pStatesList.end())
//     {
//       pStatesList.push_back(otherSubList);
//     }
//   }
// }

// std::string AmongState::toString() const noexcept
// {
//   std::string out{"{"};
//   if (pStatesList.empty())
//   {
//     out += "}";
//     return out;
//   }

//   for (auto sublist : pStatesList)
//   {
//     out += "{";
//     for (auto val : sublist)
//     {
//       out += std::to_string(val) + ", ";
//     }
//     out.pop_back();
//     out.pop_back();
//     out += "}, ";
//   }
//   out.pop_back();
//   out.pop_back();
//   out += "}";
//   return out;
// }

Among::Among(const std::string& name)
: MDDConstraint(mdd::ConstraintType::kAllDifferent, name) //,pInitialDPState(std::make_shared<Among>())
{
}

std::vector<Node*> Among::mergeNodeSelect(
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

Node* Among::mergeNodes(const std::vector<Node*>& nodesList, Arena* arena) const noexcept
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

// DPState::SPtr Among::getInitialDPState() const noexcept
// {
//   return pInitialDPState;
// }

int Among::getConstraintCountForPath( Node::EdgeList path  ) const
{
    int count = 0;
    for (int edgeIdx = 0; edgeIdx < path.size(); edgeIdx++ ) {
        Edge* edgeInPath = path[edgeIdx];

        if (std::count( getScope().begin(), getScope().end(), edgeInPath->getTail()->getVariable()) ) {
            count += 1; 
        }
    }
    return count;
}

void Among::enforceConstraintTopDown(Arena* arena, std::vector<std::vector<Node*>>& mddRepresentation) const
{
  for (int layer = 0; layer < mddRepresentation.size(); layer++) {
      bool layerInConstraint = false;
      for (auto var : getScope()) {
          if (mddRepresentation[layer][0]->getVariable()->getId() == var->getId()) {
            layerInConstraint = true;
            break;
          }
      }

      if (layerInConstraint) {
          std::unordered_map< int, Node* > nodeByConstraintCount;

          for (int nodeIdx = 0; nodeIdx < mddRepresentation[layer].size(); nodeIdx++) {
              Node* node = mddRepresentation[layer][nodeIdx];
              std::unordered_map<uint32_t, std::vector<Node::EdgeList>> nodeInPaths = node->getIncomingPaths();

              // First split and merge nodes according to their paths (state of constraint)
              for (auto inEdge: node->getInEdges()) {
                  Node::EdgeList path = nodeInPaths[inEdge->getUniqueId()][0];

                  // Count how many occurrences in a given path
                  // If different paths leads to same number of occurrences, then paths lead to same nodes and should be merged
                  // All path to same node should lead to the same count, so use the first one.
                  int count = getConstraintCountForPath(path);


                  if ( nodeByConstraintCount.find( count ) != nodeByConstraintCount.end() ) {
                    // Equivalent node exist
                    inEdge->setHead( nodeByConstraintCount[count] );
                  } else {

                    // Create new node for incoming edge
                    Node* newNode = arena->buildNode(node->getLayer(), node->getVariable());
                    nodeByConstraintCount[count] = newNode;
                    inEdge->setHead( newNode );

                    // Remove invalid values from new node
                    auto nodeDomain = *(node->getValuesMutable());
                    auto newNodeDomain = *(newNode->getValuesMutable());
                    for (int k = 0; k < newNodeDomain.size(); k++) {
                      int newNodeVal = newNodeDomain[k];
                      auto iter = std::find(nodeDomain.begin(), nodeDomain.end(), newNodeVal);
                      
                      if ( iter == nodeDomain.end()) {
                          auto newIter = std::find(newNodeDomain.begin(), newNodeDomain.end(), newNodeVal);
                          newNodeDomain.erase(newIter);
                      }
                    }


                    // Copy outgoing edges for new node
                    for (auto x : node->getOutEdges()) {
                      arena->buildEdge(newNode, x->getHead(), x->getValue(), x->getValue());
                    }
                    mddRepresentation[layer].push_back( newNode );
                  }
              }
          }

          // Splitting and merging nodes for current layer should be consistent at this point.
          //---------------------------------------------------------------------------------//

          for (int nodeIdx = 0; nodeIdx < mddRepresentation[layer].size(); nodeIdx++) {
              Node* node = mddRepresentation[layer][nodeIdx];
              for (auto outEdge : node->getOutEdges()) {
                  // If edge could create conflict, split into separate node
                  if ( std::count(pConstraintDomain.begin(), pConstraintDomain.end(), outEdge->getValue()) ) {
                    Node* head = outEdge->getHead();
                    std::unordered_map<uint32_t, std::vector<Node::EdgeList>> headInPaths = head->getIncomingPaths();

                    Node::EdgeList path = headInPaths[ head->getInEdges()[0]->getUniqueId() ][0];
                    int count = getConstraintCountForPath( path );

                    // If reached upper bound of constraint, this edge is no longer valid.
                    if (count >= pUpperBound) {
                        outEdge->removeEdgeFromNodes();
                        auto nodeDomain = *(node->getValuesMutable());
                        auto iter = std::find(nodeDomain.begin(), nodeDomain.end(), outEdge->getValue());
                        if ( iter != nodeDomain.end()) {
                            nodeDomain.erase(iter);
                        }
                        // The head of the edge could become unreachable if removed only edge leading to it
                        // If so, delete it from memory
                        if (head->getInEdges().size() == 0) {
                            arena->deleteNode( head->getUniqueId() );
                        }
                    } else {

                        Node* newNode = arena->buildNode(head->getLayer(), head->getVariable());
                        outEdge->setHead( newNode );

                        // Remove invalid values from new node
                        auto nodeDomain = *(head->getValuesMutable());
                        auto newNodeDomain = *(newNode->getValuesMutable());
                        for (int k = 0; k < newNodeDomain.size(); k++) {
                          int newNodeVal = newNodeDomain[k];
                          auto iter = std::find(nodeDomain.begin(), nodeDomain.end(), newNodeVal);
                          
                          if ( iter == nodeDomain.end()) {
                              auto newIter = std::find(newNodeDomain.begin(), newNodeDomain.end(), newNodeVal);
                              newNodeDomain.erase(newIter);
                          }
                        }

                        // Copy all outgoing edges to the new node
                        for (auto x: head->getOutEdges()) {
                            arena->buildEdge(newNode, x->getHead(), x->getValue(), x->getValue());
                        }

                        // Add new node to the mdd representation
                        mddRepresentation[newNode->getLayer()].push_back(newNode);
                    }

                  }
              }
          }
      }
  }
}



void Among::enforceConstraintBottomUp(Arena* arena, std::vector<std::vector<Node*>>& mddRepresentation) const
{
  int lastNodeLayer =  mddRepresentation.size()-1;

  std::queue<Node*> queue;

  // Go through each node in the last layer
  for ( int nodeIdx = 0; nodeIdx < mddRepresentation[lastNodeLayer].size(); nodeIdx++ ) {
      Node* node = mddRepresentation[lastNodeLayer][nodeIdx];

      std::unordered_map<int,int> countByValue;
      Edge* inEdge = node->getInEdges()[0];

      // Check count in path
      // After top-down all paths to a node should lead to the same count. 
      // So I can use any path to count.
      std::unordered_map<uint32_t, std::vector<Node::EdgeList>> inPaths = node->getIncomingPaths();
      Node::EdgeList path = inPaths[inEdge->getUniqueId()][0];
      int count = getConstraintCountForPath( path );

      // If count less than lower bound at last layer, only edges in constraint domain are possible
      // Assume that there is at least a feasible solution, otherwise leaf node will be disconnected from the rest of the graph.
      if (count < pLowerBound) {
        for (auto x : node->getOutEdges()) {
            if (std::count( pConstraintDomain.begin(), pConstraintDomain.end(), x->getValue()) == 0) {
              x->removeEdgeFromNodes();
              auto nodeDomain = *(node->getValuesMutable());
              auto iter = std::find(nodeDomain.begin(), nodeDomain.end(), x->getValue());
              if ( iter != nodeDomain.end()) {
                  nodeDomain.erase(iter);
              }
          }
        }
        queue.push( node );
      }
  }


  while (queue.size() > 0) {
     Node* curretNode = queue.front();
     queue.pop();

      // Node does not lead to a solution
     if (curretNode->getOutEdges().size() == 0) {
        // So delete node and check its parents
        for (auto x : curretNode->getInEdges()) {
           Node* parent = x->getTail();
           queue.push( parent );
           x->removeEdgeFromNodes();
        }
     }
  }

  // At this point only edges leading to feasible solutions should be left
  
}

void Among::enforceConstraint(Node* node, Arena* arena,
                                     std::vector<std::vector<Node*>>& mddRepresentation,
                                     std::vector<Node*>& newNodesList) const
{
  // Hackish way of running this only once
  // Current design calls enforce constraint for every node, but this constrain needs a global approach.
  if (node == mddRepresentation[0][0]) {
    enforceConstraintTopDown( arena, mddRepresentation);
    enforceConstraintBottomUp( arena, mddRepresentation);
  }

}

};