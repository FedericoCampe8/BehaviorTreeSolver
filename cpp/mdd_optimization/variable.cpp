#include "mdd_optimization/variable.hpp"

namespace mdd {

Variable::Variable(uint32_t id, uint32_t layer, const std::vector<int64_t>& availableValues)
: pId(id),
  pLayerIndex(layer),
  pAvailableValues(availableValues)
{
}

}  // namespace mdd
