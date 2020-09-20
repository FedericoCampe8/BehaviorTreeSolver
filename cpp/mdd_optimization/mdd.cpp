#include "mdd_optimization/mdd.hpp"

#include <stdexcept>  // for std::invalid_argument

namespace {
constexpr uint32_t kLayerZero{0};
}  // namespace

namespace mdd {

MDD::MDD(MDDProblem::SPtr problem, int32_t width)
: pMaxWidth(width),
  pProblem(problem)
{
  if (problem == nullptr)
  {
    throw std::invalid_argument("MDD - empty pointer to the problem");
  }

  if (width < 1)
  {
    throw std::invalid_argument("MDD - invalid witdh size");
  }

  // TODO: max width not implemented in constraints

  // Resize the number of layers of this MDD to have one layer per variable in the problem.
  // Note: there is one more corresponding to the terminal node layer
  pNodesPerLayer.resize(pProblem->getVariables().size() + 1);
}

void MDD::buildMDD()
{
  // Build relaxed MDD
  buildRelaxedMDD();
}


void MDD::buildRelaxedMDD()
{
  // Build the root node
  pNodesPerLayer.at(kLayerZero).push_back(std::make_unique<Node>(kLayerZero));
}

}  // namespace mdd
