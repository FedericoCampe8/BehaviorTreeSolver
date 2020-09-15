//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for a Dynamic Programming model.
// Each constraint should provide its own DP model.
//

#pragma once

#pragma once

#include <cstdint>  // for uint32_t
#include <memory>   // for std::shared_ptr

#include "system/system_export_defs.hpp"

namespace btsolver {
namespace optimization {

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
  virtual DPState::SPtr next(int32_t domainElement) const noexcept;

  /// Returns the cost of going to next state from this state
  /// given "domainElement".
  /// Returns the value of the element by default
  virtual double cost(int32_t domainElement) const noexcept;

  /// Returns whether or not this
  virtual bool isInfeasible() const noexcept;

  // Returns a string representing this state
  virtual std::string toString() const noexcept;

 protected:
  /// Returns true if this is equal to "other".
  /// Returns false otherwise
  virtual bool isEqual(const DPState* other) const noexcept;

 private:
   static uint32_t kNextID;

 private:
   /// Unique identifier for this variable
   uint32_t pStateId{0};
};


/**
 * \brief A DP model represents an optimization model
 *        based on Dynamic Programming. The model is a template class
 *        that depends on the representation of the state "State".
 *        The "State" type should provide fast operations and copies.
 *
 * The State type must provide the following operators:
 * - next(uint32_t): for the transition function;
 * - cost(uint32_t): for the cost function;
 * - isEmpty(): to return whether or not this is an empty state;
 * - operator ==: to return whether or not the two states are equal
 */
template <typename State>
class DPModel {
public:
  /// Applies the transition function from the given state and domain element
  /// to obtain the next state
  State executeTransitionFunction(const State& state, int32_t domainElement)
  {
    // Returns: "what is the next state reachable from 'state' using 'domainElement'?"
    return state.next(domainElement);
  }

  /// Applies the cost function from the given state and domain element
  /// to obtain the transition cost
  double executeTransitionCostFunction(const State& state, int32_t domainElement)
  {
    // Returns: "how much does it cost to move from 'state' using 'domainElement'?"
    return state.cost(domainElement);
  }
};

}  // namespace optimization
}  // namespace btsolver
