//
// Copyright OptiLab 2020. All rights reserved.
//
// All different constraint based on BT optimization.
//

#pragma once

#include <cstdint>  // for int32_t
#include <memory>   // for std::unique_ptr
#include <vector>

#include "mdd_optimization/edge.hpp"
#include "mdd_optimization/mdd_problem.hpp"
#include "mdd_optimization/node.hpp"
#include "mdd_optimization/variable.hpp"

#include "system/system_export_defs.hpp"

namespace mdd {

class SYS_EXPORT_CLASS MDD {
 public:
  /// List of all the (pointers to the) nodes in an MDD layer
  using NodesLayerList = std::vector<Node*>;

  /// List of all the layers of the MDD
  using MDDLayersList = std::vector<NodesLayerList>;

  using UPtr = std::unique_ptr<MDD>;

 public:
  MDD(MDDProblem::SPtr problem, int32_t width);

  void buildMDD();

  const MDDLayersList& getNodesPerLayer() const noexcept
  {
    return pNodesPerLayer;
  }

private:
  /// Max width of the MDD
  int32_t pMaxWidth{-1};

  /// Optimization model/problem to solve
  MDDProblem::SPtr pProblem{nullptr};

  /// List of all layers with nodes in this MDD
  MDDLayersList pNodesPerLayer;
};

}  // namespace mdd
