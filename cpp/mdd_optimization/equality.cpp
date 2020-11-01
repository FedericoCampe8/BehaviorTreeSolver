#include "mdd_optimization/equality.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>  // for std::invalid_argument
#include <utility>    // for std::move

// #define DEBUG

namespace mdd {

EqualityState::EqualityState()
: DPState()
{
}

EqualityState::EqualityState(const EqualityState& other)
{
  pStatesList = other.pStatesList;
}

EqualityState::EqualityState(EqualityState&& other)
{
  pStatesList = std::move(other.pStatesList);
}

EqualityState& EqualityState::operator=(const EqualityState& other)
{
  if (&other == this)
  {
    return *this;
  }

  pStatesList = other.pStatesList;
  return *this;
}

EqualityState& EqualityState::operator=(EqualityState&& other)
{
  if (&other == this)
  {
    return *this;
  }

  pStatesList = std::move(other.pStatesList);
  return *this;
}

bool EqualityState::isEqual(const DPState* other) const noexcept
{
  auto otherDPState = reinterpret_cast<const EqualityState*>(other);

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

bool EqualityState::isInfeasible() const noexcept
{
  return pStatesList.empty();
}

DPState::SPtr EqualityState::next(int64_t domainElement, DPState*) const noexcept
{
  auto state = std::make_shared<EqualityState>();
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

double EqualityState::cost(int64_t domainElement, DPState*) const noexcept
{
  return static_cast<double>(domainElement);
}

void EqualityState::mergeState(DPState* other) noexcept
{
  if (other == nullptr)
  {
    return;
  }

  auto otherDP = reinterpret_cast<const EqualityState*>(other);
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

std::string EqualityState::toString() const noexcept
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

Equality::Equality(const std::string& name)
: MDDConstraint(mdd::ConstraintType::kEquality, name),
  pInitialDPState(std::make_shared<EqualityState>())
{
}


std::vector<Node*> Equality::mergeNodeSelect(
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

Node* Equality::mergeNodes(const std::vector<Node*>& nodesList, Arena* arena) const noexcept
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

DPState::SPtr Equality::getInitialDPState() const noexcept
{
  return pInitialDPState;
}

NodeDomain Equality::getConstraintValuesForPath(const std::vector<Edge*>& path) const
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


void Equality::enforceAdmissibleValuesForLayer(int layer, std::vector<int64_t> admissibleValues, Arena* arena,
                          std::vector<std::vector<Node*>>& mddRepresentation) const
{
    std::vector<Node*> nodes = mddRepresentation[layer];
    for (int i = 0; i < nodes.size(); i++) {
        Node* node = nodes[i];
        auto outEdges = node->getOutEdges();
        for (auto edge : outEdges) {
            if ( std::count(admissibleValues.begin(), admissibleValues.end(), edge->getValue()) == 0) {
                auto head = edge->getHead();
                edge->removeEdgeFromNodes();
                arena->deleteEdge( edge->getUniqueId() );

                // If node has no outgoing edges it does not lead to solution.
                // So check its parents and see if they need to be removed
                if (node->getOutEdges().size() == 0) {
                    auto inEdges = node->getInEdges();
                    for (auto inEdge : inEdges) {
                        auto parent = inEdge->getTail();
                        inEdge->removeEdgeFromNodes();
                        arena->deleteEdge( edge->getUniqueId() );

                        if (parent->getOutEdges().size() == 0) {
                            eraseUnfeasiblePredecessors(parent, arena, mddRepresentation);
                        }
                    }
                    
                    // Delete node because it doesn't lead to a solution
                    arena->deleteNode( node->getUniqueId() );
                }
                
                // If head has no incoming edges delete successors as they cannot lead to a path
                if (head->getInEdges().size() == 0) {
                    eraseUnfeasibleSuccessors(head, arena, mddRepresentation);
                }
            }
        }
    }
}



void Equality::enforceConstraint(Arena* arena, std::vector<std::vector<Node*>>& mddRepresentation,
                                     std::vector<Node*>& newNodesList) const
{

    assert( getScope().size() == 2 );

    int id1 = getScope()[0]->getId();
    int id2 = getScope()[1]->getId();

    int firstLayerInConstraint = (id1 < id2) ? id1 : id2;
    int lastLayerInConstraint = (id1 > id2) ? id1 : id2;

    std::vector<int64_t> admissibleValues;

    for (auto val : getScope()[0]->getAvailableValues()) {

        if (std::count(getScope()[1]->getAvailableValues().begin(), 
                getScope()[1]->getAvailableValues().end(), val)) {
                    admissibleValues.push_back(val);
                }
    }


    // Go through all nodes in x1 and x2...every edge that does not have admissible value needs to be removed.
    enforceAdmissibleValuesForLayer(firstLayerInConstraint, admissibleValues, arena, mddRepresentation);
    enforceAdmissibleValuesForLayer(lastLayerInConstraint, admissibleValues, arena, mddRepresentation);
    

    std::vector<Node*> lastLayer = mddRepresentation[lastLayerInConstraint];
    spp::sparse_hash_map<int, std::vector<int>> admissibleValueForX1;
    for (auto node : lastLayer ) {
        mdd::Node::IncomingPathList incomingPaths = node->getIncomingPathsFromVarWithId(firstLayerInConstraint);
    
        std::vector<int> inValues;
        for (auto inEdge : node->getInEdges()) {
            for (auto path : incomingPaths[inEdge->getUniqueId()] ) {
               Edge* firstEdge = path[0];
               inValues.push_back( firstEdge->getValue() ); 
            }
        }

        std::vector<Edge*> outEdges = node->getOutEdges();
        bool nodeDeleted = false;

        for (auto edge : outEdges) {
            if (std::count(inValues.begin(), inValues.end(), edge->getValue()) == 0) {
                Node* head = edge->getHead();

                edge->removeEdgeFromNodes();
                arena->deleteEdge( edge->getUniqueId() );
                    
                if (head->getInEdges().size() == 0)
                {
                  eraseUnfeasibleSuccessors(head, arena, mddRepresentation);
                }

                //If node is no longer feasible, delete it and check parents
                if (node->getOutEdges().size() == 0) {
                    std::vector<Edge*> inEdges = node->getInEdges();
                    for (auto inEdge: inEdges) {
                      Node* parent = inEdge->getTail();
                      bool parentIsValid = false;
                      for (auto outEdge: parent->getOutEdges()) {
                          if (outEdge->getHead() != node) {
                            parentIsValid = true;
                            break;
                          }
                      }

                      if (parentIsValid == false) {
                        eraseUnfeasiblePredecessors(parent, arena, mddRepresentation);
                      }

                      arena->deleteEdge( inEdge->getUniqueId() );
                    }
                    mddRepresentation[node->getLayer()].erase( std::find(mddRepresentation[node->getLayer()].begin(), 
                        mddRepresentation[node->getLayer()].end(), node) );
                    
                    arena->deleteNode( node->getUniqueId() );
                    nodeDeleted = true;
                }

            }
        }


        if (nodeDeleted == false) {
          //After clearing invalid values for x2, keep track of valid ones for x1
          for (auto inEdge : node->getInEdges()) {
              for (auto path : incomingPaths[inEdge->getUniqueId()] ) {
                Edge* firstEdge = path[0];
                Node* x1 = firstEdge->getTail();
                for (auto outEdge : node->getOutEdges()) {
                  admissibleValueForX1[ x1->getUniqueId() ].push_back( outEdge->getValue() );
                }
              }
          }
        }
    }

    // At this point only feasible values at x2 should be left.
    // Now we need to remove invalid edges from x1.

    // At this point only there should only be edges from x1 where x2 can have equality
    std::vector<Node*> x1nodes = mddRepresentation[firstLayerInConstraint];
    spp::sparse_hash_map<int, int> valueByNodeId; 
    spp::sparse_hash_map<int, Node*> nodeByNodeId; 

    //Keeping track what nodes are splits of which nodes
    spp::sparse_hash_map<int, int> mappedNodesIds;

    
    int totalNodes = x1nodes.size();
    for (int i = 0; i < totalNodes; i++) {
        Node* node = x1nodes[i];
        std::vector<int> validValues = admissibleValueForX1[node->getUniqueId()];
        
        std::vector<Edge*> outEdges = node->getOutEdges();
        for (int k = 0; k < outEdges.size(); k++) {
            Edge* edge = outEdges[k];
            Node* head = edge->getHead();
            if (std::count(validValues.begin(), validValues.end(), edge->getValue()) == 0) {
                // Invalid value from x1
                edge->removeEdgeFromNodes();
                    
                if (head->getInEdges().size() == 0)
                {
                  eraseUnfeasibleSuccessors(head, arena, mddRepresentation);
                }

                arena->deleteEdge( edge->getUniqueId() );

                //If node is no longer feasible, delete it and check parents
                if (node->getOutEdges().size() == 0) {
                    std::vector<Edge*> inEdges = node->getInEdges();
                    for (auto inEdge: inEdges) {
                      Node* parent = inEdge->getTail();
                      bool parentIsValid = false;
                      for (auto outEdge: parent->getOutEdges()) {
                          if (outEdge->getHead() != node) {
                            parentIsValid = true;
                            break;
                          }
                      }

                      if (parentIsValid == false) {
                        eraseUnfeasiblePredecessors(parent, arena, mddRepresentation);
                      }

                      arena->deleteEdge( inEdge->getUniqueId() );
                    }
                    mddRepresentation[node->getLayer()].erase( std::find(mddRepresentation[node->getLayer()].begin(), 
                        mddRepresentation[node->getLayer()].end(), node) );
                    
                    arena->deleteNode( node->getUniqueId() );
                }
            } else {
              // Valid value from x1
              int headId = head->getUniqueId();
              if (valueByNodeId.find(headId) != valueByNodeId.end() ) {
                    if (valueByNodeId[headId] == edge->getValue()) {
                        edge->setHead( nodeByNodeId[headId] );
                    } else {
                        Node* newHead = arena->buildNode(head->getLayer(), head->getVariable());

                        for (auto headOutEdge : head->getOutEdges()) {
                            arena->buildEdge(newHead, headOutEdge->getHead(), edge->getDomainLowerBound(), edge->getDomainUpperBound());
                        }
                        edge->setHead( newHead );
                        mddRepresentation[newHead->getLayer()].push_back( newHead );

                        mappedNodesIds[ newHead->getUniqueId() ] = head->getUniqueId();
                        
                        nodeByNodeId[ newHead->getUniqueId() ] = newHead;

                    }

                } else {
                    valueByNodeId[headId] = edge->getValue();
                    nodeByNodeId[headId] = head;
                }
            } 
        }
    }


    // After nodes in x1 are split, each node that was split needs to take a different path.
    // So split successors until you reach nodes in x2.
    for (int layer = firstLayerInConstraint+1; layer < lastLayerInConstraint-1; layer++) {
        for (int nodeIdx = 0; nodeIdx < mddRepresentation[layer].size(); nodeIdx++) {
            Node* node = mddRepresentation[layer][nodeIdx];
            //If node is a copy, create a copy for the head.
            //A copy is needed to differentiate unique solutions
            if ( mappedNodesIds.find(node->getUniqueId()) != mappedNodesIds.end() ) {

                int mappedId = mappedNodesIds[node->getUniqueId()];
                Node* mappedNode = nodeByNodeId[mappedId];
                
                for (auto outEdge : mappedNode->getOutEdges()) {
                    Node* head = outEdge->getHead();

                    if (mappedNodesIds.find(head->getUniqueId()) != mappedNodesIds.end()) {
                        int mappedHeadId = mappedNodesIds[head->getUniqueId()];
                        Node* mappedHead = nodeByNodeId[mappedHeadId];
                        outEdge->setHead( mappedHead );
                    } else {
                        Node* mappedHead = arena->buildNode(head->getLayer(), head->getVariable());

                        for (auto headOutEdge: head->getOutEdges()) {
                            arena->buildEdge(mappedHead, headOutEdge->getHead(), headOutEdge->getDomainLowerBound(), headOutEdge->getDomainUpperBound());
                        }
                        mddRepresentation[mappedHead->getLayer()].push_back( mappedHead );
                        mappedNodesIds[ mappedHead->getUniqueId() ] = head->getUniqueId();
                        nodeByNodeId[ mappedHead->getUniqueId() ] = mappedHead;
                    }
                }

            }
        }
    }


}



};
