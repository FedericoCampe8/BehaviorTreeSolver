#include <thrust/find.h>

#include "State.cuh"

__host__ __device__
DP::State::State(unsigned int variablesCount, std::byte* storage) :
    cost(0),
    selectedValues(variablesCount, storage),
    admissibleValues(variablesCount, selectedValues.storageEnd())
{}

__host__ __device__
bool DP::State::isAdmissible(unsigned int value) const
{
    return thrust::find(thrust::seq, admissibleValues.begin(), admissibleValues.end(), static_cast<uint8_t>(value)) != admissibleValues.end();
}
__host__ __device__
bool DP::State::isSelected(unsigned int value) const
{
    return thrust::find(thrust::seq, selectedValues.begin(), selectedValues.end(), static_cast<uint8_t>(value)) != selectedValues.end();
}
__host__ __device__
DP::State& DP::State::operator=(const DP::State& other)
{
    cost = other.cost;
    selectedValues = other.selectedValues;
    admissibleValues = other.admissibleValues;
    return *this;
}

__host__ __device__
std::size_t DP::State::sizeOfStorage(unsigned int variablesCount)
{
    return StaticVector<uint8_t>::sizeOfStorage(variablesCount) * 2;
}
