#include "data_structures/int_set.hpp"

#include <cassert>

namespace ds {

IntSet::IntSet()
: pEnd(static_cast<int64_t>(pSet.npos))
{
}

IntSet::IntSet(int64_t min, int64_t max, bool filled)
: pEnd(static_cast<int64_t>(pSet.npos))
{
  resize(min, max, filled);
}

IntSet::IntSet(const IntSet& other)
{
  pEnd = other.pEnd;
  pMin = other.pMin;
  pMax = other.pMax;
  pSet = other.pSet;
}

IntSet::IntSet(IntSet&& other)
{
  pEnd = other.pEnd;
  pMin = other.pMin;
  pMax = other.pMax;
  pSet = other.pSet;
  other.pSet.clear();
}

IntSet& IntSet::operator=(const IntSet& rhs)
{
  assert(rhs.pMax == pMax && rhs.pMin == pMin);
  if (this != &rhs)
  {
    pSet = rhs.pSet;
  }
  return *this;
}

IntSet& IntSet::operator=(IntSet&& rhs)
{
  assert(rhs.pMax == pMax && rhs.pMin == pMin);
  if (this != &rhs)
  {
    pSet = rhs.pSet;
    rhs.pSet.clear();
  }
  return *this;
}


void IntSet::resize(int64_t min, int64_t max, bool filled)
{
  assert(min == 0);
  pMin = min;
  pMax = max;

  //shift = (-1) * _min;
  pSet.resize(pMax - pMin + 1);
  if(filled)
  {
    pSet.set();
  }
  else
  {
    pSet.reset();
  }
}

bool IntSet::contains(int64_t elem) const
{
  assert(elem >= pMin && elem <= pMax);
  return(pSet.test(elem));
}

void IntSet::add(int64_t elem)
{
  assert(elem >= pMin && elem <= pMax);
  pSet.set(elem, true);
}

void IntSet::remove(int64_t elem)
{
  assert(elem >= pMin && elem <= pMax);
  pSet.set(elem, false);
}

int64_t IntSet::getFirst() const
{
  return static_cast<int64_t>((pSet.find_first()));
}

int64_t IntSet::getNext(int64_t elem)
{
  assert(elem >= pMin && elem <= pMax);
  for (auto v{elem+1}; v <= pMax; v++)
  {
    if (pSet.test(v))
    {
      return v;
    }
  }
  return pEnd;
}

int64_t IntSet::getEnd() const
{
  return pEnd;
}

void IntSet::clear()
{
  return pSet.clear();
}

uint64_t IntSet::getSize() const noexcept
{
  return pSet.count();
}

void IntSet::addAllElements()
{
  pSet.set();
}

bool IntSet::operator ==(const IntSet& intset) const
{
    return this->pSet == intset.pSet;
}

bool IntSet::operator <(const IntSet& intset) const
{
    return this->pSet < intset.pSet;
}

}  // namespace ds
