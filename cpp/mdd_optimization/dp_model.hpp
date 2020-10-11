//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for a Dynamic Programming model.
//

#pragma once

#include <cstdint>  // for uint32_t
#include <limits>   // for std::numeric_limits
#include <memory>   // for std::shared_ptr
#include <string>
#include <utility>  // for std::pair
#include <vector>

#include "system/system_export_defs.hpp"

namespace mdd {

/**
 * \brief The base class for a Dynamic Programming model's state.
 */
class SYS_EXPORT_STRUCT DPState {
 public:
  using ReplacementNodeList = std::vector<std::pair<uint32_t, int64_t>>;

  using UPtr = std::unique_ptr<DPState>;
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

  /**
   * \brief reset this state to the default state
   */
  virtual void resetState() noexcept;

  /**
   * \brief clones this states and returns a pointer to the clone.
   */
  virtual DPState* clone() const noexcept;

  /**
   * \brief returns the path values (edges) up to this state.
   * \note this DOES NOT work for merged nodes.
   */
  const std::vector<int64_t>& cumulativePath() const noexcept { return pPath; }

  /// Merges "other" into this DP State
  virtual void mergeState(DPState* other) noexcept;

  /// Returns true if this DP state is merged.
  /// Returns false otherwise
  virtual bool isMerged() const noexcept { return false; }

  /// Checks (hopefully quickly) if the given element would produce an infeasible
  /// next state given the current state
  virtual bool isValueFeasible(int64_t domainElement) const noexcept;

  /**
   * \brief updates this state to the next state in the DP transition function
   *        obtained by applying "val" to "state"
   */
  virtual void updateState(DPState* state, int64_t val);

  /**
   * \brief returns the cost of taking the given value.
   * \note return +INF if the value is inducing a non-admissible state
   */
  virtual double getCostPerValue(int64_t value);

  /**
   * \brief returns the list of pairs <cost, value> that can be obtains
   *        from this state when following an edge with value in [lb, ub].
   * \note values that are higher than or equal the given incumbet are discarded.
   */
  virtual std::vector<std::pair<double, int64_t>> getCostListPerValue(
          int64_t lb, int64_t ub, double incumbent);

  /// Returns the list of "width" feasible states that can be reached from the current
  /// DP state using values in [lb, ub].
  /// It also returns, as last element of the vector, the state representing all
  /// other states that could have been taken from the current state but discarded
  /// due to maximum width.
  /// @note Returns an empty vector if no state is reachible from the current one.
  /// @note Excludes all states that have a cost greater than or equal to the given incumbent
  virtual std::vector<DPState::SPtr> next(int64_t lb, int64_t ub, uint64_t width,
                                          double incumbent) const noexcept;

  /**
   * \brief Updates the list "nextStateList" feasible states that can be reached from the current
   *        DP state using values in [lb, ub].
   *        It excludes states that have a value higher or equal to the given incumbent.
   *        Returns the list of pairs <index_of_replaced_state, edge_value>.
   *        The index of replaced node is the index on the input vector "nextStateList".
   */
  virtual ReplacementNodeList next(int64_t lb, int64_t ub, double incumbent,
                                   std::vector<DPState::UPtr>* nextStateList) const noexcept;

  /// Returns the next state reachable from this state given "domainElement".
  /// Returns self by default.
  /// @note nextDPState is used on some constraints during the bottom-up pass to calculate
  ///       next state based on the information of the state on the next node calculated
  ///       during top-down procedure
  virtual DPState::SPtr next(int64_t domainElement, DPState* nextDPState=nullptr) const noexcept;

  /// Returns the cost of going to next state from this state
  /// given "domainElement".
  /// Returns the value of the element by default.
  /// @note some costs on the current state can be calculated only if
  /// the previous state arriving to the current state is known
  virtual double cost(int64_t domainElement, DPState* fromState=nullptr) const noexcept;

  /// Returns the cumulative cost up to this state
  virtual double cumulativeCost() const noexcept;

  /// Returns whether or not this
  virtual bool isInfeasible() const noexcept;

  /// Returns a string representing this state
  virtual std::string toString() const noexcept;

  /// Set state a non-default according to given flag
  void setNonDefaultState(bool isDefault=false) noexcept { pIsDefault = isDefault; }

  /// Returns whether or not this is a default state
  bool isDefaultState() const noexcept { return pIsDefault; }

  /// Returns true if this is equal to "other".
  /// Returns false otherwise
  virtual bool isEqual(const DPState* other) const noexcept;

 protected:
  /// Flag indicating whether or not this state works on top-down or bottom-up filtering
  bool pTopDownFiltering{true};

  /// Cumulative path found up to this state
  std::vector<int64_t> pPath;

  /// Cost of the path up to this state, i.e., cumulative cost
  double pCost{std::numeric_limits<double>::max()};

  /**
   * \brief copy over the base state.
   */
  void copyBaseDPState(DPState* other) const;

 private:
   static uint32_t kNextID;

 private:
   /// Unique identifier for this variable
   uint32_t pStateId{0};

   /// Flag indicating default (or not) state
   bool pIsDefault{true};
};

}  // namespace mdd
