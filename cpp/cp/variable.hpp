//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for a CP variable.
//

#pragma once

#include <cstdint>    // for uint32_t
#include <memory>  // for std::shared_ptr
#include <string>

#include "cp/domain.hpp"
#include "cp/bitmap_domain.hpp"
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
  using FiniteDomain = Domain<BitmapDomain>;

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

  /// Returns the raw pointer to the internal domain.
  /// Note: use it only if you know what you are doing!
  FiniteDomain* getDomainMutable() const noexcept { return pDomain.get(); }

  /// Returns true if this variable is ground (i.e., labeled).
  /// Returns false otherwise
  bool isGround() const noexcept { return (pDomain->size() == 1); }

  /// Returns the value of this variable if ground, otherwise returns the
  /// lower bound element
  int32_t getValue() const noexcept;

  /// Returns true if the domain of this variable is empty.
  /// Returns false otherwise
  bool isInvalid() const noexcept { return pDomain->isEmpty(); }

private:
  static uint32_t kNextID;

private:
  /// Unique identifier for this variable
  uint32_t pVariableId{0};

  /// This variable's name
  const std::string pName{};

  /// Variable's domain
  std::shared_ptr<FiniteDomain> pDomain;
};

}  // namespace cp
}  // namespace btsolver
