#include "cp/bitmap_domain.hpp"

#include <limits>  // for std::numeric_limits


namespace btsolver {
namespace cp {

BitmapDomain::BitmapDomain(int32_t lowerBound,  int32_t upperBound)
{
  configure(lowerBound, upperBound);
}

void BitmapDomain::configure(int32_t lowerBound,  int32_t upperBound)
{
  pLowerBound = lowerBound;
  pUpperBound = upperBound;
  pBitset = boost::dynamic_bitset<>(upperBound - lowerBound + 1);
  pBitset.set();
}

int32_t BitmapDomain::min() const noexcept
{
  if (empty())
  {
    return std::numeric_limits<int32_t>::min();
  }
  return pLowerBound + getFirst();
}

int32_t BitmapDomain::max() const noexcept
{
  if (empty())
  {
    return std::numeric_limits<int32_t>::max();
  }

  std::size_t pos{0};
  while (pBitset.find_next(pos) != pBitset.npos)
  {
    ++pos;
  }
  return pLowerBound + static_cast<int32_t>(pos);
}

bool BitmapDomain::contains(int32_t d) const noexcept
{
  if (d < pLowerBound || d > pUpperBound)
  {
    return false;
  }

  // Get the bit position in the bitmap considering the offset and test
  // the bit in the bitmap
  return pBitset.test(d - pLowerBound);
}

void BitmapDomain::remove(int32_t d) noexcept
{
  if (d < pLowerBound || d > pUpperBound)
  {
    return;
  }
  pBitset.reset(d - pLowerBound);
}

void BitmapDomain::reinsertElement(int32_t d) noexcept
{
  if (d < pLowerBound || d > pUpperBound)
  {
    return;
  }
  pBitset.set(d - pLowerBound);
}

}  // namespace cp
}  // namespace btsolver
