//
// Copyright OptiLab 2020. All rights reserved.
//
// All different constraint based on BT optimization.
//

#pragma once

#include <cstdint>  // for int64_t
#include <limits>   // for std::numeric_limits
#include <memory>   // for std::unique_ptr
#include <vector>

#include "system/system_export_defs.hpp"

namespace mdd {

class SYS_EXPORT_CLASS Variable {
 public:
  using UPtr = std::unique_ptr<Variable>;
  using SPtr = std::shared_ptr<Variable>;

 public:
  Variable(uint32_t id, uint32_t layer, int64_t lowerBound, int64_t upperBound);
  Variable(uint32_t id, uint32_t layer, const std::vector<int64_t> &availableValues);

  /// Return this variable's unique identifier
  uint32_t getId() const noexcept { return pId; }

  /// Returns this variable' domain
  const std::vector<int64_t>& getAvailableValues() const noexcept { return pAvailableValues;}

  /// Returns this variable's lower bound
  int64_t getLowerBound() const noexcept { return pLowerBound; }

  /// Returns this variable's upper bound
  int64_t getUpperBound() const noexcept { return pUpperBound; }

 private:
  /// Variable unique identifier
  uint32_t pId;

  /// MDD Layer this variable is at
  uint32_t pLayerIndex;

  /// This variable's domain
  std::vector<int64_t> pAvailableValues;

  /// Lower bound on this variable
  int64_t pLowerBound { std::numeric_limits<int64_t>::min() };

  /// Upper bound on this variable
  int64_t pUpperBound { std::numeric_limits<int64_t>::max() };
};

}  // namespace mdd
