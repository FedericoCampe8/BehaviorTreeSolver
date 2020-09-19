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

#include <sparsepp/spp.h>

#include "bt/node_status.hpp"
#include "system/system_export_defs.hpp"

// Forward declarations
namespace btsolver {
class BehaviorTreeArena;
class Edge;
}  // namespace btsolver

namespace btsolver {

enum class NodeType {
  ConditionState = 0,
  Log,
  OptimizationRunner,
  Selector,
  Sequence,
  State,
  UndefinedType
};

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
  using NodeCallback = std::function<void()>;
  using NodeRunCallback = std::function<NodeStatus()>;
  using EdgeList = std::vector<Edge*>;
  using UPtr = std::unique_ptr<Node>;
  using SPtr = std::shared_ptr<Node>;

public:
  /**
   * \brief Node constructor:
   *        - name: name of this node. This should be a unique name
   *        - arena: the memory/map of nodes in the behavior tree
   */
  Node(const std::string& name, NodeType nodeType, BehaviorTreeArena* arena);
  virtual ~Node();

  /// Returns the cast'ed version of the given node
  template<class Type>
  Type* cast() noexcept { return reinterpret_cast<Type*>(this); }

  /// Returns this node's unique identifier
  uint32_t getUniqueId() const noexcept { return pNodeId; }

  /// Returns this node's name
  const std::string& getName() const noexcept { return pNodeName; }

  /// Returns the result of this node
  NodeStatus getResult() const noexcept { return pResult; }

  /// Returns the type of this node
  NodeType getNodeType() const noexcept { return pNodeType; }

  /// Registers the run callback: the function to call on run
  void registerRunCallback(NodeRunCallback&& runCB) noexcept { pRunCallback = runCB; }

  /// Registers the configure callback: the function to call on enter
  void registerConfigureCallback(NodeCallback&& configCB) noexcept { pConfigureCallback = configCB; }

  /// Registers the cleanup callback: the function to call on exit
  void registerCleanupCallback(NodeCallback&& cleanupCB) noexcept { pCleanupCallback = cleanupCB; }

  /// Registers the cancel callback: the function to call when canceled
  void registerCancelCallback(NodeCallback&& cancelCB) noexcept { pCancelCallback = cancelCB; }

  /// Adds an incoming edge to this node
  void addIncomingEdge(Edge* edge);

  /// Adds an outgoing edges to this node
  void addOutgoingEdge(Edge* edge);

  /// Remove an incoming edge from this node
  void removeIncomingEdge(Edge* edge);

  /// Remove an outgoing edge from this node
  void removeOutgoingEdge(Edge* edge);

  /// Returns the (usually only) incoming edge
  Edge* getIncomingEdge() const noexcept;

  /// Returns the list of all incoming edges
  const EdgeList& getAllIncomingEdges() const noexcept { return pIncomingEdges; }

  /// Returns the list of all outgoing edges
  const EdgeList& getAllOutgoingEdges() const noexcept { return pOutgoingEdges; }

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
  /// Identifiers of incoming edges.
  /// Note: usually there is only one incoming edge,
  /// i.e., one parent for each node
  EdgeList pIncomingEdges;

  /// Identifiers of outgoing edges
  EdgeList pOutgoingEdges;

  /// Sets storing incoming/outgoing edges for quick lookup
  spp::sparse_hash_set<uint32_t> pIncomingEdgeSet;
  spp::sparse_hash_set<uint32_t> pOutgoingEdgeSet;

  /// Returns this node's arena
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

  /// Type of this node
  NodeType pNodeType{NodeType::UndefinedType};

  /// Status/result of this node
  NodeStatus pResult{NodeStatus::kPending};

  /// Forced status
  NodeStatus pForcedState{NodeStatus::kUndefined};

  /// Callback
  NodeRunCallback pRunCallback;
  NodeCallback pConfigureCallback;
  NodeCallback pCleanupCallback;
  NodeCallback pCancelCallback;
};

}  // namespace btsolver
