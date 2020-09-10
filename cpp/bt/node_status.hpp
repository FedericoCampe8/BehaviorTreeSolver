//
// Copyright OptiLab 2020. All rights reserved.
//
// Struct enumerating behavior status.
//

#pragma once

#include <memory>
#include <string>

#include "system/system_export_defs.hpp"

namespace btsolver {

class SYS_EXPORT_CLASS NodeStatus {
public:
  enum class NodeStatusType {
    // Node is in pre-run state
    kPending = 0,
    // Node is running
    kActive = 1,
    // Node has completed successfully
    kSuccess = 2,
    // Node has completed with errors
    kFail = 3,
    // Node has been canceled
    kCancel = 4,
    // Undefined state
    kUndefined = 100
  };

  using SPtr = std::shared_ptr<NodeStatus>;

public:
  NodeStatus(NodeStatusType status = NodeStatusType::kPending);
  ~NodeStatus() = default;

  /// Overrides the internal status
  void changeStatus(NodeStatusType status) noexcept;

  /// Merge status: if the merged status has a greater value than the current status,
  /// the merged status is used
  void merge(NodeStatusType status) noexcept;

  /// Returns this node's internal status type
  NodeStatusType getStatus() const noexcept { return pStatusType; }

  /// Returns the string representation of this NodeStatus
  std::string toString() const noexcept;

private:
  NodeStatusType pStatusType{NodeStatusType::kPending};

};

}  // namespace btsolver
