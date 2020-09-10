//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for Blackboard data-structure used in Behavior Trees.
// The blackboard allows nodes to share information.
//

#pragma once

#include <cstdint>  // for uint32_t
#include <memory>
#include <string>

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
  Blackboard() = default;
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

private:
  using Memory = spp::sparse_hash_map<std::string, BlackboardValue>;

private:
  /// The "memory" of this blackboard
  Memory pMemory;

  /// Map of nodes (unique ids) and their status
  NodeStatusMap pNodeStatusMap;

};

}  // namespace btsolver
