#include "bt/node.hpp"

#include <stdexcept>  // for std::runtime_error

namespace btsolver {

// Initialize unique Identifier for nodes
uint32_t Node::kNextID = 0;

Node::Node(const std::string& name,
           Blackboard::SPtr blackboard,
           CallbackRunSPtr runCallback,
           CallbackSPtr configureCallback,
           CallbackSPtr cleanupCallback,
           CallbackSPtr cancelCallback)
: pNodeId(Node::kNextID++),
  pNodeName(name),
  pBlackboard(blackboard ? blackboard : std::make_shared<Blackboard>()),
  pRunCallback(runCallback),
  pConfigureCallback(configureCallback),
  pCleanupCallback(cleanupCallback),
  pCancelCallback(cancelCallback)
{
}

void Node::configure()
{
  if (pConfigureCallback)
  {
    // Run the user provided callback
    (*pConfigureCallback)(pBlackboard);
  }

  // Change the status to active
  pResult.changeStatus(NodeStatus::NodeStatusType::kActive);
}

void Node::cleanup()
{
  if (pResult.getStatus() == NodeStatus::NodeStatusType::kActive)
  {
    this->cancel();
  }

  if (pCleanupCallback)
  {
    // Run the user provided callback
    (*pCleanupCallback)(pBlackboard);
  }

  // Reset the status to pending
  pForcedState.changeStatus(NodeStatus::NodeStatusType::kUndefined);
  pResult.changeStatus(NodeStatus::NodeStatusType::kPending);
}

void Node::run() {
  if (pForcedState.getStatus() != NodeStatus::NodeStatusType::kUndefined)
  {
    // Force the status
    pResult.changeStatus(pForcedState.getStatus());
  }
  else if (pRunCallback)
  {
    auto status = (*pRunCallback)(pBlackboard);
    pResult.changeStatus(status);
  }
  else
  {
    throw std::runtime_error("Node - run: callback not found");
  }
}

void Node::cancel()
{
  pForcedState.changeStatus(NodeStatus::NodeStatusType::kCancel);
  if (pCancelCallback)
  {
    // Run the user provided callback
    (*pCancelCallback)(pBlackboard);
  }

  // Run the node after canceling
  this->run();
}

void Node::force(NodeStatus::NodeStatusType status)
{
  pForcedState.changeStatus(status);
}

void Node::tick()
{
  if (pResult.getStatus() == NodeStatus::NodeStatusType::kPending)
  {
    this->configure();
  }

  // Run the node
  this->run();

  const auto postRunResult = pResult.getStatus();
  if (!(postRunResult == NodeStatus::NodeStatusType::kActive ||
          postRunResult == NodeStatus::NodeStatusType::kPending))
  {
    // The node is not active anymore, execute the cleanup
    this->cleanup();
  }
}

}  // namespace btsolver

