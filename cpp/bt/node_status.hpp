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

enum class NodeStatus {
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

SYS_EXPORT_FCN std::string statusToString(NodeStatus state) noexcept;

}  // namespace btsolver
