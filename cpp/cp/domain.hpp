//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for the domain of a variable.
// This is a template class to avoid too many
// virtual pointer re-directions at run time.
// Note: this are all FINITE domains.
//

#pragma once

#include <cstdint>  // for uint32_t
#include <stdexcept>  // for std::invalid_argument

namespace btsolver {
namespace cp {

/**
 * \brief Iterator class for domains.
 *        Iterates over the elements of the domain.
 */
template <typename BaseDomain>
class DomainIterator {
public:
  DomainIterator() = default;

  DomainIterator(BaseDomain* domain)
  : pDomain(domain)
  {
    if (pDomain == nullptr)
    {
      throw std::invalid_argument("DomainIterator - empty domain pointer");
    }
  }

  /// Returns whether or not the domain is empty
  bool empty() const noexcept
  {
    return pDomain->empty();
  }

  /// Reset the iterator to the first element (if any)
  void reset() noexcept
  {
    pElementPointer = 0;
  }

  /// Returns whether or not the iterator is at the last domain element
  bool atEnd() noexcept
  {
    return value() == pDomain->max();
  }

  /// Returns the current domain value
  int32_t value() noexcept
  {
    while (!pDomain->containsElementAtPosition(pElementPointer))
    {
      ++pElementPointer;
    }

    return pDomain->getOriginalLowerBound() + pElementPointer;
  }

  /// Move the iterator to the next domain element
  void moveToNext() noexcept
  {
    ++pElementPointer;
  }

private:
  /// Pointer to the actual domain implementation
  BaseDomain* pDomain{nullptr};

  /// Pointer to the current element
  uint32_t pElementPointer{0};
};

/**
 * \brief The Domain class representing a variable's domain.
 *        This is a wrapper around a BaseDomain which, in turn,
 *        is the actual implementation of the domain and it should
 *        provide, at least, the following methods:\
 *        - configure(d1, d2): setup the domain as [d1, d2]
 *        - contains(d): returns whether or not the domain contains d integer value
 */
template <typename BaseDomain>
class Domain {
 public:
  /**
   * \brief Creates a domain [lowerBound, upperBound].
   *        Throws std::invalid_argument if upperBound < lowerBound
   */
  Domain(int32_t lowerBound,  int32_t upperBound)
  : pLowerBound(lowerBound),
    pUpperBound(upperBound)
  {
    if (upperBound < lowerBound)
    {
      throw std::invalid_argument("Domain - invalid domain bounds");
    }

    // Configure the internal domain representation
    resetDomain();
  }

  /// Resets this domain to its original content
  void resetDomain()
  {
    pDomain.configure(pLowerBound, pUpperBound);
    pIterator = DomainIterator<BaseDomain>(&pDomain);
  }

  /// Sets this domain to empty
  void setToEmpty() noexcept { pDomain.setEmpty(); }

  /// Returns whether or not this domain is empty
  bool isEmpty() const noexcept { return pDomain.empty(); }

  /// Returns the min element of the domain
  uint32_t minElement() const noexcept { return pDomain.min(); }

  /// Returns the max element of the domain
  uint32_t maxElement() const noexcept { return pDomain.max(); }

  /// Returns the number of elements in the domain
  uint32_t size() const noexcept { return pDomain.size(); }

  /// Returns whether or not this domain contains the element 'd'
  bool contains(int32_t d) const noexcept
  {
    return pDomain.contains(d);
  }

  /// Removes 'd' from this domain (if 'd' is contained in this domain)
  void removeElement(int32_t d) noexcept { pDomain.remove(d); }

  DomainIterator<BaseDomain>& getIterator() noexcept { return pIterator; }

 private:
  /// Actual domain implementation
  BaseDomain pDomain{};

  /// Domain iterator
  DomainIterator<BaseDomain> pIterator;

  // Original domain bounds
  int32_t pLowerBound{0};
  int32_t pUpperBound{0};
};

}  // namespace cp
}  // namespace btsolver
