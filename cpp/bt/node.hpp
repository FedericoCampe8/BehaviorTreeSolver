//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for Behavior Tree nodes.
// This class defined the standard callback and data structures
// to execute the nodes.
//

#pragma once

#include <cstdint>     // for uint32_t
#include <functional>  // for std::function
#include <limits>      // for std::numeric_limits
#include <memory>      // for std::unique_ptr
#include <string>
#include <vector>

#include "bt/blackboard.hpp"
#include "bt/node_status.hpp"
#include "system/system_export_defs.hpp"

// Forward declarations
namespace btsolver {
class BehaviorTreeArena;
}  // namespace btsolver

namespace btsolver {

/**
 * \brief Main class for BT nodes.
 * Callback follow a set state machine on each tick:
 * PENDING -> CONFIGURE -> ACTIVE <-> RUN -> CANCEL ------>+
 *                                     |                   |
 *                                     +-> SUCCESS/FAIL -->+-> CLEANUP
 * The data structure is provided by a Blackboard.
 * By default, each node creates its own Blackboard, however,
 * it is possible to specify the blackboard as argument to allow nodes
 * to share data.
 */
class SYS_EXPORT_CLASS Node {
public:
  /// Node callback methods
  using NodeCallback = std::function<void(Blackboard*)>;
  using NodeRunCallback = std::function<NodeStatus(Blackboard*)>;
  using UPtr = std::unique_ptr<Node>;
  using SPtr = std::shared_ptr<Node>;

public:
  /**
   * \brief Node constructor:
   *        - name: name of this node. This should be a unique name
   *        - arena: the memory/map of nodes in the behavior tree
   *        - blackboard: blackboard for this node. If nullptr is passed, a default one is created.
   */
  Node(const std::string& name,
       BehaviorTreeArena* arena,
       Blackboard* blackboard=nullptr);

  virtual ~Node();

  /// Returns this node's unique identifier
  uint32_t getUniqueId() const noexcept { return pNodeId; }

  /// Returns this node's name
  const std::string& getName() const noexcept { return pNodeName; }

  /// Returns the result of this node
  NodeStatus getResult() const noexcept { return pResult; }

  /// Returns this node's status
  NodeStatus getStatus() noexcept { return pBlackboard->getNodeStatus(getUniqueId()); }

  /// Returns the raw pointer to this node's blackboard
  Blackboard* getBlackboardMutable() const noexcept { return pBlackboard; }

  /// Registers the run callback: the function to call on run
  void registerRunCallback(NodeRunCallback&& runCB) noexcept { pRunCallback = runCB; }

  /// Registers the configure callback: the function to call on enter
  void registerConfigureCallback(NodeCallback&& configCB) noexcept { pConfigureCallback = configCB; }

  /// Registers the cleanup callback: the function to call on exit
  void registerCleanupCallback(NodeCallback&& cleanupCB) noexcept { pCleanupCallback = cleanupCB; }

  /// Registers the cancel callback: the function to call when canceled
  void registerCancelCallback(NodeCallback&& cancelCB) noexcept { pCancelCallback = cancelCB; }

  /// Sets the blackboard.
  /// Note: this node does NOT take ownership of the given blackboard
  void setBlackboard(Blackboard* blackboard) noexcept { pBlackboard = blackboard; }

  /// Adds edges to this node
  void addIncomingEdge(uint32_t edgeId) { pIncomingEdges.push_back(edgeId); }
  void addOutgoingEdge(uint32_t edgeId) { pOutgoingEdges.push_back(edgeId); }

  /// Remove edges to this node
  void removeIncomingEdge(uint32_t edgeId);
  void removeOutgoingEdge(uint32_t edgeId);

  /// Returns the (usually only) incoming edge
  uint32_t getIncomingEdge() const {
    return pIncomingEdges.empty() ? std::numeric_limits<uint32_t>::max() : pIncomingEdges.front(); }

  /// Executes the node's state machine and returns the result of the run.
  /// In Behavior Tree terminology, executes a "tick"
  NodeStatus tick();

  /// Forces an execution state
  void force(NodeStatus status) noexcept { pForcedState = status; }

  /// Configuration performed once before run.
  /// This is usually used to set up internal variables
  void configure();

  /// Evaluates the current node and returns the state after run
  NodeStatus run();

  /// Cleanup performed once run returns a termination value.
  /// This is usually used to reset internal variables
  virtual void cleanup();

  /// Forces the current state to CANCEL and calls the cancel
  /// callback, if any
  virtual void cancel();

protected:
  BehaviorTreeArena* getArena() const noexcept { return pArena; }

private:
  static uint32_t kNextID;

private:
  /// Unique identifier for this node
  uint32_t pNodeId{0};

  /// This node's name
  std::string pNodeName{};

  /// Pointer to the behavior tree arena
  BehaviorTreeArena* pArena{nullptr};

  /// Status/result of this node
  NodeStatus pResult{NodeStatus::kPending};

  /// Forced status
  NodeStatus pForcedState{NodeStatus::kUndefined};

  /// Blackboard memory of this node
  Blackboard* pBlackboard{nullptr};

  /// Callback
  NodeRunCallback pRunCallback;
  NodeCallback pConfigureCallback;
  NodeCallback pCleanupCallback;
  NodeCallback pCancelCallback;

  /// Identifiers of incoming edges.
  /// Note: usually there is only one incoming edge,
  /// i.e., one parent for each node
  std::vector<uint32_t> pIncomingEdges;

  /// Identifiers of outgoing edges
  std::vector<uint32_t> pOutgoingEdges;
};

}  // namespace btsolver
