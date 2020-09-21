//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for a Dynamic Programming model.
//

#pragma once

#include <cstdint>  // for uint32_t
#include <memory>   // for std::shared_ptr

#include "system/system_export_defs.hpp"

namespace mdd {

/**
 * \brief The base class for a Dynamic Programming model's state.
 */
class SYS_EXPORT_STRUCT DPState {
 public:
  using SPtr = std::shared_ptr<DPState>;

 public:
  DPState();
  virtual ~DPState() = default;

  // Equality operator
  bool operator==(const DPState& other);

  /// Returns the unique identifier for this DP state
  uint32_t getUniqueId() const noexcept { return pStateId; }

  /// Returns the next state reachable from this state given "domainElement".
  /// Returns self by default
  virtual DPState::SPtr next(int64_t domainElement) const noexcept;

  /// Returns the cost of going to next state from this state
  /// given "domainElement".
  /// Returns the value of the element by default
  virtual double cost(int64_t domainElement) const noexcept;

  /// Returns whether or not this
  virtual bool isInfeasible() const noexcept;

  // Returns a string representing this state
  virtual std::string toString() const noexcept;

  /// Returns true if this is equal to "other".
  /// Returns false otherwise
  virtual bool isEqual(const DPState* other) const noexcept;

 private:
   static uint32_t kNextID;

 private:
   /// Unique identifier for this variable
   uint32_t pStateId{0};
};

}  // namespace mdd
