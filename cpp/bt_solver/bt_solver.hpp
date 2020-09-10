//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for the Behavior-Tree-based solver.
//

#pragma once

#include <memory>  // for std::unique_ptr

#include "bt/behavior_tree.hpp"
#include "cp/model.hpp"
#include "system/system_export_defs.hpp"

namespace btsolver {

class SYS_EXPORT_CLASS BTSolver {
 public:
  using UPtr = std::unique_ptr<BTSolver>;
  using SPtr = std::shared_ptr<BTSolver>;

 public:
  BTSolver() = default;
  ~BTSolver() = default;

  void setModel()

  /// Builds and returns a relaxed BT
  BehaviorTree::SPtr buildRelaxedBT();
};

}  // namespace btsolver
