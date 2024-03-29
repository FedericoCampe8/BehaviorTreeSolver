#include "mdd_optimization/variable.hpp"

#include <algorithm>  // for std::sort
#include <numeric>    // for std::iota
#include <stdexcept>  // for std::invalid_argument

namespace mdd {

Variable::Variable(uint32_t id, uint32_t layer, int64_t lowerBound, int64_t upperBound,
                   const std::string& name)
: pId(id),
  pLayerIndex(layer),
  pName(name),
  pLowerBound(lowerBound),
  pUpperBound(upperBound)
{
  if (pLowerBound > pUpperBound)
  {
    throw std::invalid_argument("Variable - invalid bounds");
  }
  pAvailableValues.resize(upperBound - lowerBound + 1);
  std::iota(std::begin(pAvailableValues), std::end(pAvailableValues), lowerBound);
}

Variable::Variable(uint32_t id, uint32_t layer, const std::vector<int64_t>& availableValues,
                   const std::string& name)
: pId(id),
  pLayerIndex(layer),
  pName(name),
  pAvailableValues(availableValues)
{
  if (pAvailableValues.empty())
  {
    throw std::invalid_argument("Variable - empty domain");
  }

  // Sort the list of elements
  std::sort(pAvailableValues.begin(), pAvailableValues.end());
  pLowerBound = pAvailableValues.front();
  pUpperBound = pAvailableValues.back();
}

}  // namespace mdd
