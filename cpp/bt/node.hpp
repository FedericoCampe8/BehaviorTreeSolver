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
#include <memory>
#include <string>

#include "bt/blackboard.hpp"
#include "bt/node_status.hpp"
#include "system/system_export_defs.hpp"


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
  using NodeCallback = std::function<void(const Blackboard::SPtr&)>;
  using RunCallback = std::function<NodeStatus::NodeStatusType(const Blackboard::SPtr&)>;
  using CallbackSPtr = std::shared_ptr<NodeCallback>;
  using CallbackRunSPtr = std::shared_ptr<RunCallback>;
  using UPtr = std::unique_ptr<Node>;
  using SPtr = std::shared_ptr<Node>;

public:
  /**
   * \brief Node constructor:
   *        - name: name of this node. This should be a unique name
   *        - blackboard: blackboard for this node. If nullptr is passed, a default one is created.
   *        - runCallback: the function to call on run
   *        - configureCallback: the function to call on enter
   *        - cleanupCallback: the function to call on exit
   *        - cancelCallback: the function to call when canceled
   */
  Node(const std::string& name,
       Blackboard::SPtr blackboard=nullptr,
       CallbackRunSPtr runCallback=nullptr,
       CallbackSPtr configureCallback=nullptr,
       CallbackSPtr cleanupCallback=nullptr,
       CallbackSPtr cancelCallback=nullptr);
  virtual ~Node() = default;

  /// Returns this node's unique identifier
  uint32_t getUniqueId() const noexcept { return pNodeId; }

  /// Returns this node's name
  const std::string& getName() const noexcept { return pNodeName; }

  /// Returns the result of this node
  const NodeStatus& getResult() const noexcept { return pResult; }

  /// Returns this node's data/blackboard
  Blackboard::SPtr getNodeData() const noexcept { return pBlackboard; }

  /// Registers the run callback
  void registerRunCallback(CallbackRunSPtr runCB) noexcept { pRunCallback = runCB; }

  /// Registers the configure callback
  void registerConfigureCallback(CallbackSPtr configCB) noexcept { pConfigureCallback = configCB; }

  /// Registers the cleanup callback
  void registerCleanupCallback(CallbackSPtr cleanupCB) noexcept { pCleanupCallback = cleanupCB; }

  /// Registers the cancel callback
  void registerCancelCallback(CallbackSPtr cancelCB) noexcept { pCancelCallback = cancelCB; }

  /// Sets the blackboard
  void setBlackboard(Blackboard::SPtr blackboard) noexcept { pBlackboard = blackboard; }

  /// Returns the blackboard registered within this node
  Blackboard::SPtr getBlackboard() const noexcept { return pBlackboard; }

  /// Runs the node.
  /// In Behavior Tree terminology, executes a "tick"
  void tick();

  /// Forces an execution state
  void force(NodeStatus::NodeStatusType status);

  /// Configuration performed once before run.
  /// This is usually used to set up internal variables
  void configure();

  /// Evaluates the current node
  void run();

  /// Cleanup performed once run returns a termination value.
  /// This is usually used to reset internal variables
  virtual void cleanup();

  /// Forces the current state to CANCEL and calls the cancel
  /// callback, if any
  virtual void cancel();

private:
  static uint32_t kNextID;

private:
  uint32_t pNodeId{0};
  std::string pNodeName{};

  /// Status/result of this node
  NodeStatus pResult{};

  /// Forced status
  NodeStatus pForcedState{NodeStatus::NodeStatusType::kUndefined};

  /// Blackboard memory of this node
  Blackboard::SPtr pBlackboard;

  /// Callback
  CallbackRunSPtr pRunCallback;
  CallbackSPtr pConfigureCallback;
  CallbackSPtr pCleanupCallback;
  CallbackSPtr pCancelCallback;
};

}  // namespace btsolver
