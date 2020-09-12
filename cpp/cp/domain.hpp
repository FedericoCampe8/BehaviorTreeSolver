//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for the domain of a variable.
// This is a template class to avoid too many
// virtual pointer re-directions at run time.
// Note: this are all FINITE domains.
//

#pragma once

#include <algorithm>  // for std::sort
#include <cstdint>    // for uint32_t
#include <limits>     // for std::numeric_limits
#include <memory>     // for std::unique_ptr
#include <stdexcept>  // for std::invalid_argument
#include <utility>    // for std::move
#include <vector>

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
  using UPtr = std::unique_ptr<Domain<BaseDomain>>;

 public:
  /**
   * \brief Creates a singleton domain.
   *        Throws std::invalid_argument if upperBound < lowerBound
   */
  Domain(int32_t value)
  : pLowerBound(value),
    pUpperBound(value)
  {
    // Configure the internal domain representation
    resetDomain();
  }

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

  /**
   * \brief Creates a domain over the given list of elements.
   *        Throws std::invalid_argument if the list is empty.
   */
  Domain(std::vector<int32_t>& elementList)
  {
    if (elementList.empty())
    {
      throw std::invalid_argument("Domain - empty list");
    }

    //std::sort(elementList.begin(), elementList.end());
    pLowerBound = elementList.front();
    pUpperBound = elementList.back();
    resetDomain();

    // Set this domain to empty
    pDomain.setEmptyAndPreserveBounds();

    for (auto d : elementList)
    {
      pDomain.reinsertElement(d);
    }
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

  /// Returns true if this domain is equal to the given domain.
  /// Returns false otherwise
  bool isEqual(Domain<BaseDomain>* other) const noexcept
  {
    if (other == nullptr || (this->minElement() != other->minElement()) ||
            (this->maxElement() != this->maxElement()))
    {
      return false;
    }

    // Bounds are the same.
    // The two domains are the equal iff they have the same size
    return this->size() == other->size();
  }

  /// Returns the min element of the domain
  int32_t minElement() const noexcept {
    return isEmpty() ? std::numeric_limits<uint32_t>::min() : pDomain.min(); }

  /// Returns the max element of the domain
  int32_t maxElement() const noexcept {
    return isEmpty() ? std::numeric_limits<uint32_t>::max() : pDomain.max(); }

  /// Returns the number of elements in the domain
  uint32_t size() const noexcept { return pDomain.size(); }

  /// Returns the list of all elements in this domain.
  /// Note: this method resets the iterator
  std::vector<int32_t> getElementList() noexcept
  {
    std::vector<int32_t> elementList;
    elementList.reserve(this->size());
    bool breakLoop{false};
    while(true)
    {
      if (pIterator.atEnd())
      {
        breakLoop = true;
      }

      elementList.push_back(pIterator.value());

      if (breakLoop)
      {
        pIterator.reset();
        return elementList;
      }

      pIterator.moveToNext();
    }
  }

  /// Returns whether or not this domain contains the element 'd'
  bool contains(int32_t d) const noexcept
  {
    return pDomain.contains(d);
  }

  /// Subtracts "dom" from this domain and returns the difference as a new domain
  Domain<BaseDomain>::UPtr subtract(Domain<BaseDomain>* dom)
  {
    if (this->isEmpty())
    {
      auto res = std::make_unique<Domain<BaseDomain>>(0, 0);
      res->setToEmpty();
      return std::move(res);
    }

    auto res = std::make_unique<Domain<BaseDomain>>(pLowerBound, pUpperBound);
    res->pDomain = pDomain;
    if ((this->maxElement() < dom->minElement()) ||
            (dom->maxElement() < this->minElement()))
    {
      // No intersection between domains, return this domain
      return std::move(res);
    }

    bool breakLoop{false};
    while (true)
    {
      if (res->pIterator.atEnd())
      {
        breakLoop = true;
      }

      const auto val = res->pIterator.value();
      if (!dom->contains(val))
      {
        res->pDomain.remove(val);
      }

      if (breakLoop)
      {
        break;
      }

      res->pIterator.moveToNext();
    }
    res->pIterator.reset();

    // Return the difference domain
    return std::move(res);
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
