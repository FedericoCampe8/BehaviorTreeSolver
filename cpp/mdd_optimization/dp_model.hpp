//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for a Dynamic Programming model.
//

#pragma once

#include <cstdint>  // for uint32_t
#include <memory>   // for std::shared_ptr
#include <string>

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

  /// Sets the state for top-down filtering mode
  void setStateForTopDownFiltering(bool isTopDown) { pTopDownFiltering = isTopDown; }

  /// Returns whether or not this state is used on top-down or bottom-up filtering
  bool isStateSetForTopDownFiltering() const noexcept { return pTopDownFiltering; }

  /// Merges "other" into this DP State
  virtual void mergeState(DPState* other) noexcept;

  /// Returns true if this DP state is merged.
  /// Returns false otherwise
  virtual bool isMerged() const noexcept { return false; }

  /// Returns the next state reachable from this state given "domainElement".
  /// Returns self by default.
  /// @note nextDPState is used on some constraints during the bottom-up pass to calculate
  ///       next state based on the information of the state on the next node calculated
  ///       during top-down procedure
  virtual DPState::SPtr next(int64_t domainElement, DPState* nextDPState=nullptr) const noexcept;

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

 protected:
  /// Flag indicating whether or not this state works on top-down or bottom-up filtering
  bool pTopDownFiltering{true};

 private:
   static uint32_t kNextID;

 private:
   /// Unique identifier for this variable
   uint32_t pStateId{0};
};

}  // namespace mdd
