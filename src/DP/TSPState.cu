#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "TSPState.cuh"

__host__ __device__
DP::TSPState::TSPState(unsigned int height, std::byte* storage) :
    type(Type::NotActive),
    admissibleValues(height, storage)
{}

__device__
bool DP::TSPState::isActive(TSPState const & state)
{
    return state.type == Type::Active;
}

__host__ __device__
std::size_t DP::TSPState::sizeofStorage(unsigned int capacity)
{
    return StaticVector<int32_t>::sizeofStorage(capacity);
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
    this->type = other.type;
    this->cost = other.cost;
    this->lastValue = other.lastValue;
    this->admissibleValues = other.admissibleValues;
    return *this;
}
__device__
void DP::TSPState::reset(TSPState& state)
{
    state.type = Type::NotActive;
    state.admissibleValues.clear();
}
