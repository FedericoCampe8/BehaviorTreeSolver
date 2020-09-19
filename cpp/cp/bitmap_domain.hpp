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
  BitmapDomain(int32_t lowerBound,  int32_t upperBound);

  ~BitmapDomain() = default;

  /// Returns original lower bound
  int32_t getOriginalLowerBound() const noexcept { return pLowerBound; }

  /// Returns original upper bound
  int32_t getOriginalUpperBound() const noexcept { return pUpperBound; }

  /// Comparison operators "=="
  bool operator == (const BitmapDomain& intset)
  {
    return this->pBitset == intset.pBitset;
  }

  /// Comparison operators "<"
  bool operator < (const BitmapDomain& intset)
  {
    return this->pBitset < intset.pBitset;
  }

  /// Returns the minimum element in the domain
  int32_t min() const noexcept;

  /// Returns the maximum element in the domain
  int32_t max() const noexcept;

  /// Get first element of the set
  int32_t getFirst() const noexcept { return pBitset.find_first(); }

  /// Set the domain to [lowerBound, upperBound]
  void configure(int32_t lowerBound,  int32_t upperBound);

  /// Add an element to the set
  void add(int32_t elem);

  /// Add all possible elements to the set
  void addAllElements();

  /// Get next element higher than the one passed as parameter
  int32_t getNext(int32_t elem);

  /// Sets this domain to be empty
  void setEmpty() noexcept { clear(); }

  /// Clear set
  void clear() noexcept { pBitset.reset(); }

  /// Sets this domain to be empty WITHOUT changing the bounds of this domain
  void setEmptyAndPreserveBounds() noexcept { pBitset.reset(); }

  /// Returns whether or not this domain is empty
  bool empty() const noexcept { return size() == 0; }

  /// Returns the size of this domain
  std::size_t size() const noexcept { return pBitset.count(); }

  /// Returns true if this domain contains 'd', returns false otherwise
  bool contains(int32_t d) const noexcept;

  /// Removes the element 'd', from this domain
  void remove(int32_t d) noexcept;

  /// Re-insert the given element in the domain
  void reinsertElement(int32_t d) noexcept;

  /// Returns true if the element at given position is set to true.
  /// Returns false otherwise
  bool containsElementAtPosition(uint32_t pos) const noexcept { return pBitset.test(pos); }

  /// Intersect with another domain
  void intersectWith(BitmapDomain& intset);

  /// Set minus operation
  void setMin(BitmapDomain& intset);

private:
  /// Original domain bounds
  int32_t pLowerBound{0};
  int32_t pUpperBound{0};

  /// Domain size
  int32_t pSize{0};

  /// Bitset data structure for contiguous elements
  boost::dynamic_bitset<> pBitset;
};

}  // namespace cp
}  // namespace btsolver
