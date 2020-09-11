//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for Blackboard data-structure used in Behavior Trees.
// The blackboard allows nodes to share information.
//

#pragma once

#include <atomic>
#include <cstdint>  // for uint32_t
#include <memory>
#include <string>
#include <vector>

#include <sparsepp/spp.h>

#include "bt/node_status.hpp"
#include "system/system_export_defs.hpp"


namespace btsolver {

/**
 * \brief A value that can be stored in the Blackboard memory.
 */
struct SYS_EXPORT_STRUCT BlackboardValue {
};

class SYS_EXPORT_CLASS Blackboard {
public:
  using NodeStatusMap = spp::sparse_hash_map<uint32_t, NodeStatus>;
  using SPtr = std::shared_ptr<Blackboard>;

public:
  Blackboard();
  ~Blackboard() = default;

  /// Saves the given <key, value> pair in the Blackboard
  void save(const std::string& key, const BlackboardValue& value)
  {
    pMemory[key] = value;
  }

  /// Returns the value stored in this blackboard given its key
  const BlackboardValue& get(const std::string& key) const
  {
    return pMemory.at(key);
  }

  BlackboardValue& get(const std::string& key)
  {
    return pMemory[key];
  }

  /// Returns all of the node status
  const NodeStatusMap& getStatus() const noexcept { return pNodeStatusMap; }

  /// Returns the status of a specific node
  NodeStatus getNodeStatus(uint32_t nodeId) noexcept;

  /// Sets the status of a specific node
  void setNodeStatus(uint32_t nodeId, NodeStatus status)
  {
    pNodeStatusMap[nodeId] = status;
  }

  /// Clears all the internal status
  void clearNodeStatus() { pNodeStatusMap.clear(); }

  /// Adds a new (CP) node state to the blackboard.
  /// The state will be set active/non-active according to the given value
  void addState(uint32_t stateId, bool isActive=true);

  /// Returns the value (active/non-active) of the given state
  bool checkState(uint32_t stateId) const noexcept;

  /// Gets the value of the given state (active/non-active).
  /// Sets the state to non-active (regardless), and returns
  /// the value of the state before it was set to non-active
  bool checkAndDeactivateState(uint32_t stateId);

  std::vector<uint32_t>& getMostRecentStatesList() noexcept { return pMostRecentStatesList; }

private:
  using Memory = spp::sparse_hash_map<std::string, BlackboardValue>;
  using StateMemory = spp::sparse_hash_map<uint32_t, bool>;

private:
  /// The "memory" of this blackboard
  Memory pMemory;

  /// Map of nodes (unique ids) and their status
  NodeStatusMap pNodeStatusMap;

  /// Map of active/non-active states.
  /// This map is mainly used by CP solver to activate states
  StateMemory pStateMemory;

  /// This is a performance 'trick'.
  /// This vector is used to store the new states created by each child
  /// while building the exact BT. This way, a new child doesn't have
  /// to traverse all the left sub-tree to obtain them but it can
  /// read them directly from here.
  /// Of course, this vector must be updated at each iteration of
  /// the split-filtering algorithm
  std::vector<uint32_t> pMostRecentStatesList;

};

}  // namespace btsolver
