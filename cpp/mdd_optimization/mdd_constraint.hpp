//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for an MDD constraint.
//

#pragma once

#include <deque>
#include <cstdint>  // for uint32_t
#include <memory>   // for std::shared_ptr


#include "mdd_optimization/arena.hpp"
#include "mdd_optimization/constraint.hpp"
#include "mdd_optimization/dp_model.hpp"
#include "mdd_optimization/node.hpp"

#include "system/system_export_defs.hpp"

namespace mdd {

/**
 * \brief Class representing a constraint used with Behavior Trees solving.
 *        Constraints used with BT solvers need to provide the corresponding
 *        Dynamic Programming model.
 */
class SYS_EXPORT_CLASS MDDConstraint : public Constraint {
 public:
   using UPtr = std::unique_ptr<MDDConstraint>;
   using SPtr = std::shared_ptr<MDDConstraint>;

 public:
   MDDConstraint(ConstraintType type, const std::string& name="");

   /// Returns true if this constraint needs to run a bottom-up pass on the mdd.
   /// Returns false otherwise
   virtual bool runsBottomUp() const noexcept { return false; }

   /// Sets this constraint for bottom-up separation
   virtual void setForBottomUpFiltering() noexcept {}

   /// Sets this constraint for top-down separation
   virtual void setForTopDownFiltering() noexcept {}

   /// Applies some heuristics to select a subset of nodes in the given layer to merge
   virtual std::vector<Node*> mergeNodeSelect(
           int layer, const std::vector<std::vector<Node*>>& mddRepresentation) const noexcept = 0;

   /// Merges the given list of nodes and returns the representative merged node
   virtual Node* mergeNodes(const std::vector<Node*>& nodesList, Arena* arena) const noexcept = 0;

   /**
    * \brief Enforces this constraint on the given MDD node.
    * \arg node: the node on which to enforce this constraint
    * \arg arena: node and edge builder
    * \arg mddRepresentation: full MDD representation (layer by layer)
    * \arg newNodesList: list of all the new nodes created on the same level
    */
   virtual void enforceConstraint(Arena* arena,
                                  std::vector<std::vector<Node*>>& mddRepresentation,
                                  std::vector<Node*>& newNodesList) const = 0;

   /// Returns the initial state of the DP transformation chain
   virtual DPState::SPtr getInitialDPState() const noexcept = 0;

   void eraseUnfeasibleSuccessors(Node* node, Arena* arena, std::vector<std::vector<Node*>>& mddRepresentation) const;
   void eraseUnfeasiblePredecessors(Node* node, Arena* arena, std::vector<std::vector<Node*>>& mddRepresentation) const;

};

}  // namespace mdd
