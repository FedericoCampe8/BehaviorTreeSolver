//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for a CP variable.
//

#pragma once

#include <cstdint>  // for uint32_t
#include <limits>   // for std::numeric_limits
#include <memory>   // for std::shared_ptr
#include <string>

#include "system/system_export_defs.hpp"

namespace btsolver {
namespace cp {

/**
 * \brief The base class for a CP finite-domain variable
 */
class SYS_EXPORT_CLASS Variable {
public:
  using UPtr = std::unique_ptr<Variable>;
  using SPtr = std::shared_ptr<Variable>;

public:
  /**
   * \brief Constructs a new variable given its name (possibly unique) and
   *        the domain as [lowerBound, upperBound].
   */
  Variable(const std::string& name, int32_t lowerBound, int32_t upperBound);

  /// Returns this variable's unique identifier
  uint32_t getUniqueId() const noexcept { return pVariableId; }

  /// Returns this variable's name
  const std::string& getName() const noexcept { return pName; }

  /// Returns true if this variable is ground (i.e., labeled).
  /// Returns false otherwise
  bool isGround() const noexcept { return pLowerBound == pUpperBound; }

  /// Returns the value of this variable if ground, otherwise returns the
  /// lower bound element
  int32_t getValue() const noexcept {  return pLowerBound; }

  /// Returns the original lower bound
  int32_t getLowerBound() const noexcept { return pLowerBound; }

  /// Returns the original upper bound
  int32_t getUpperBound() const noexcept { return pUpperBound; }

private:
  static uint32_t kNextID;

private:
  /// Unique identifier for this variable
  uint32_t pVariableId{0};

  /// This variable's name
  const std::string pName{};

  /// Lower bound on this variable
  int32_t pLowerBound{std::numeric_limits<int32_t>::min()};

  /// Upper bound on this variable
  int32_t pUpperBound{std::numeric_limits<int32_t>::max()};
};

}  // namespace cp
}  // namespace btsolver
