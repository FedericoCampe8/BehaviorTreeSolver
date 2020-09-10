//
// Copyright OptiLab 2020. All rights reserved.
//
// Bitmap implementation of a domain.
//

#pragma once

#include <cstdio>   // for std::size_t
#include <cstdint>  // for uint32_t

#include <boost/dynamic_bitset.hpp>

#include "system/system_export_defs.hpp"

namespace btsolver {
namespace cp {

class SYS_EXPORT_CLASS BitmapDomain {
public:
  BitmapDomain() = default;
  ~BitmapDomain() = default;

  /// Return original bounds
  int32_t getOriginalLowerBound() const noexcept { return pLowerBound; }
  int32_t getOriginalUpperBound() const noexcept { return pUpperBound; }

  /// Returns the minimum element in the domain
  int32_t min() const noexcept;

  /// Returns the maximum element in the domain
  int32_t max() const noexcept;

  /// Set the domain to [lowerBound, upperBound]
  void configure(int32_t lowerBound,  int32_t upperBound);

  /// Sets this domain to be empty
  void setEmpty() noexcept { pBitset.reset(); }

  /// Returns whether or not this domain is empty
  bool empty() const noexcept { return size() == 0; }

  /// Returns the size of this domain
  std::size_t size() const noexcept { return pBitset.count(); }

  /// Returns true if this domain contains 'd', returns false otherwise
  bool contains(int32_t d) const noexcept;

  /// Removes the element 'd', from this domain
  void remove(int32_t d) noexcept;

  /// Returns true if the element at given position is set to true.
  /// Returns false otherwise
  bool containsElementAtPosition(uint32_t pos) const noexcept { return pBitset.test(pos); }

private:
  /// Bitset data structure for contiguous elements
  boost::dynamic_bitset<> pBitset;

  // Original domain bounds
  int32_t pLowerBound{0};
  int32_t pUpperBound{0};
};

}  // namespace cp
}  // namespace btsolver
