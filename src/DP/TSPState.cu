#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "TSPState.cuh"

__host__ __device__
DP::TSPState::TSPState(unsigned int height, std::byte* storage) :
    active(false),
    exact(true),
    admissibleValues(height, storage)
{}

__host__ __device__
std::size_t DP::TSPState::sizeofStorage(unsigned int height)
{
    return StaticVector<int32_t>::sizeofStorage(height);
}

__host__ __device__
bool DP::TSPState::isAdmissible(int value) const
{
    return thrust::binary_search(thrust::seq, admissibleValues.begin(), admissibleValues.end(), value);
}

__host__ __device__
void DP::TSPState::addToAdmissibles(int value)
{
    assert(not isAdmissible(value));

    admissibleValues.pushBack(value);
    thrust::sort(thrust::seq, admissibleValues.begin(),admissibleValues.end());
}

__device__
DP::TSPState& DP::TSPState::operator=(TSPState const & other)
{
    this->active = other.active;
    this->exact = other.exact;
    this->cost = other.cost;
    this->lastValue = other.lastValue;
    this->admissibleValues = other.admissibleValues;
    return *this;
}

__device__
void DP::TSPState::reset(TSPState& state)
{
    state.active = false;
    state.exact = true;
    state.admissibleValues.clear();
}
