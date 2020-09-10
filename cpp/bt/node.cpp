#include "bt/node.hpp"

#include <stdexcept>  // for std::runtime_error

namespace btsolver {

// Initialize unique Identifier for nodes
uint32_t Node::kNextID = 0;

Node::Node(const std::string& name, Blackboard::SPtr blackboard)
: pNodeId(Node::kNextID++),
  pNodeName(name),
  pBlackboard(blackboard ? blackboard : std::make_shared<Blackboard>())
{
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

}  // namespace btsolver

