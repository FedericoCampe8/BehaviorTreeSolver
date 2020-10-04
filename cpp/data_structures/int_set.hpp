//
// Copyright OptiLab 2020. All rights reserved.
//
// Data structure to store positive sets of integers.
//

#pragma once

#include <cstdint>  // for int64_t
#include <memory>

#include <boost/dynamic_bitset.hpp>

#include "system/system_export_defs.hpp"

namespace ds {

class SYS_EXPORT_CLASS IntSet final {
 public:
  using UPtr = std::unique_ptr<IntSet>;
  using SPtr = std::shared_ptr<IntSet>;

 public:
  IntSet();
  IntSet(int64_t min, int64_t max, bool filled);

  IntSet(const IntSet& other);
  IntSet(IntSet&& other);

  IntSet& operator=(const IntSet& rhs);
  IntSet& operator=(IntSet&& rhs);

  /// Resizes the set
  void resize(int64_t min, int64_t max, bool filled);

  /// Returns true if the set contains the element.
  /// Returns false otherwise
  bool contains(int64_t elem) const;

  /// Adds an element to the set
  void add(int64_t elem);

  /// Adds all elements to the set
  void addAllElements();

  /// Remove the given element, if contained
  void remove(int64_t elem);

  /// Returns the size of this set
  uint64_t getSize() const noexcept;

  /// Returns the fist element of the set
  int64_t getFirst() const;

  /// Returns the next element higher than the one passed as parameter
  int64_t getNext(int64_t elem);

  /// Returns the end of the set (beyond last element)
  int64_t getEnd() const;

  /// Clears the set
  void clear();

  /// Fills all elements between min and max
  void fill();

  /// Comparison operators
  bool operator ==(const IntSet& intset) const;
  bool operator <(const IntSet& intset) const;

 private:

  /// Actual set
  boost::dynamic_bitset<> pSet;

  /// position beyond end of the set
  int64_t pEnd{0};

  /// Number of elements in the set
  uint64_t pSize{0};

  /// Minimum possible element of the set
  int64_t pMin{-1};

  /// Maximum possible element of the set
  int64_t pMax{0};
};

}  // namespace ds
