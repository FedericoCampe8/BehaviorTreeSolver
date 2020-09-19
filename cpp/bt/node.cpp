#include "bt/node.hpp"

#include <algorithm>  // for std::find
#include <cassert>
#include <stdexcept>  // for std::runtime_error

#include "bt/behavior_tree_arena.hpp"

namespace btsolver {

// Initialize unique Identifier for nodes
uint32_t Node::kNextID = 0;

Node::Node(const std::string& name, NodeType nodeType, BehaviorTreeArena* arena)
: pNodeId(Node::kNextID++),
  pNodeName(name),
  pArena(arena),
  pNodeType(nodeType)
{
  if (pArena == nullptr)
  {
    throw std::invalid_argument("Node: empty pointer to the arena");
  }
}

Node::~Node()
{
  // Remove this node from all edges
  auto inEdgeListCopy = pIncomingEdges;
  for (auto inEdge : inEdgeListCopy)
  {
    if (inEdge != nullptr)
    {
      inEdge->removeTail();
    }
  }

  auto outEdgeListCopy = pOutgoingEdges;
  for (auto outEdge : outEdgeListCopy)
  {
    if (outEdge != nullptr)
    {
      outEdge->removeHead();
    }
  }
}

void Node::configure()
{
  if (pConfigureCallback)
  {
    // Run the user provided callback
    pConfigureCallback();
  }

  // Change the status to active
  pResult = NodeStatus::kActive;
}

void Node::cleanup()
{
  if (pResult == NodeStatus::kActive)
  {
    this->cancel();
  }

  if (pCleanupCallback)
  {
    // Run the user provided callback
    pCleanupCallback();
  }

  // Reset the status to pending
  pForcedState = NodeStatus::kUndefined;
  pResult = NodeStatus::kPending;
}

NodeStatus Node::run() {
  if (pForcedState != NodeStatus::kUndefined)
  {
    // Force the status
    pResult = pForcedState;
  }
  else if (pRunCallback)
  {
    pResult = pRunCallback();
  }
  else
  {
    throw std::runtime_error("Node - run: callback not found");
  }
  return pResult;
}

void Node::cancel()
{
  pForcedState = NodeStatus::kCancel;
  if (pCancelCallback)
  {
    // Run the user provided callback
    pCancelCallback();
  }

  // Run the node after canceling
  this->run();
}

NodeStatus Node::tick()
{
  if (pResult == NodeStatus::kPending)
  {
    this->configure();
  }

  // Run the node
  const auto result = this->run();
  if (!(result == NodeStatus::kActive || result == NodeStatus::kPending))
  {
    // The node is not active anymore, execute the cleanup.
    // Note: cleanup has side-effects on the internal state of this node
    this->cleanup();
  }
  return result;
}

void Node::addIncomingEdge(Edge* edge)
{
  if (edge == nullptr || (pIncomingEdgeSet.find(edge->getUniqueId()) != pIncomingEdgeSet.end()))
  {
    return;
  }
  pIncomingEdges.push_back(edge);
  pIncomingEdgeSet.insert(edge->getUniqueId());

  // Set this node as the tail of the given edge
  edge->setTail(this);
}

void Node::addOutgoingEdge(Edge* edge)
{
  if (edge == nullptr || (pOutgoingEdgeSet.find(edge->getUniqueId()) != pOutgoingEdgeSet.end()))
  {
    return;
  }
  pOutgoingEdges.push_back(edge);
  pOutgoingEdgeSet.insert(edge->getUniqueId());

  // Set this node as the head of the given edge
  edge->setHead(this);
}

Edge* Node::getIncomingEdge() const noexcept
{
  if (pIncomingEdges.empty())
  {
    return nullptr;
  }
  return pIncomingEdges.at(0);
}

void Node::removeIncomingEdge(Edge* edge)
{
  if (edge == nullptr || (pIncomingEdgeSet.find(edge->getUniqueId()) == pIncomingEdgeSet.end()))
  {
    return;
  }

  auto iter = std::find(pIncomingEdges.begin(), pIncomingEdges.end(), edge);
  assert(iter != pIncomingEdges.end());
  pIncomingEdges.erase(iter);
  pIncomingEdgeSet.erase(edge->getUniqueId());

  // Update the edge
  edge->removeTail();
}

void Node::removeOutgoingEdge(Edge* edge)
{

  if (edge == nullptr || (pOutgoingEdgeSet.find(edge->getUniqueId()) == pOutgoingEdgeSet.end()))
  {
    return;
  }

  auto iter = std::find(pOutgoingEdges.begin(), pOutgoingEdges.end(), edge);
  assert(iter != pOutgoingEdges.end());
  pOutgoingEdges.erase(iter);
  pOutgoingEdgeSet.erase(edge->getUniqueId());

  // Update the edge
  edge->removeHead();
}

}  // namespace btsolver
