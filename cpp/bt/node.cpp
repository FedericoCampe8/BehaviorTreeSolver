#include "bt/node.hpp"

#include <algorithm>  // for std::find
#include <stdexcept>  // for std::runtime_error

#include "bt/behavior_tree_arena.hpp"

namespace {
constexpr uint32_t kDefaultNumEdges{100};
}  // namespace

namespace btsolver {

// Initialize unique Identifier for nodes
uint32_t Node::kNextID = 0;

Node::Node(const std::string& name,
           BehaviorTreeArena* arena,
           Blackboard* blackboard)
: pNodeId(Node::kNextID++),
  pNodeName(name),
  pArena(arena),
  pBlackboard(blackboard)
{
  if (pArena == nullptr)
  {
    throw std::invalid_argument("Node: empty pointer to the arena");
  }

  if (pBlackboard == nullptr)
  {
    throw std::invalid_argument("Node: empty pointer to the blackboard");
  }

  pIncomingEdges.reserve(10);
  pOutgoingEdges.reserve(kDefaultNumEdges);
}

Node::~Node()
{
  // Remove this node from the edges
  for (auto inEdge : pIncomingEdges)
  {
    auto edge = pArena->getEdge(inEdge);
    edge->resetTail();
  }

  for (auto inEdge : pOutgoingEdges)
  {
    auto edge = pArena->getEdge(inEdge);
    edge->resetHead();
  }
}

void Node::configure()
{
  if (pConfigureCallback)
  {
    // Run the user provided callback
    pConfigureCallback(pBlackboard);
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
    pCleanupCallback(pBlackboard);
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
    pResult = pRunCallback(pBlackboard);
  }
  else
  {
    throw std::runtime_error("Node - run: callback not found");
  }
  pBlackboard->setNodeStatus(getUniqueId(), pResult);
  return pResult;
}

void Node::cancel()
{
  pForcedState = NodeStatus::kCancel;
  if (pCancelCallback)
  {
    // Run the user provided callback
    pCancelCallback(pBlackboard);
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

void Node::removeIncomingEdge(uint32_t edgeId)
{
  auto iter = std::find(pIncomingEdges.begin(), pIncomingEdges.end(), edgeId);
  if (iter != pIncomingEdges.end())
  {
    pIncomingEdges.erase(iter);
  }
}

void Node::removeOutgoingEdge(uint32_t edgeId)
{
  auto iter = std::find(pOutgoingEdges.begin(), pOutgoingEdges.end(), edgeId);
  if (iter != pOutgoingEdges.end())
  {
    pOutgoingEdges.erase(iter);
  }
}

}  // namespace btsolver
