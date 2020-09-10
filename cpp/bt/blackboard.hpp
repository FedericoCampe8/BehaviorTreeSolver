//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for Blackboard data-structure used in Behavior Trees.
//

#pragma once

#include <memory>

#include "system/system_export_defs.hpp"


namespace btsolver {

class SYS_EXPORT_CLASS Blackboard {
public:
  using SPtr = std::shared_ptr<Blackboard>;

public:
  Blackboard() = default;
  ~Blackboard() = default;
};

}  // namespace btsolver
