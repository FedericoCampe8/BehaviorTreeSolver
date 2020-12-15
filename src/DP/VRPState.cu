#include <thrust/find.h>

#include "VRPState.cuh"

__host__ __device__
DP::VRPState::VRPState(unsigned int variablesCount, std::byte* storage) :
    cost(0),
    selectedValues(variablesCount, storage),
    admissibleValues(variablesCount, selectedValues.storageEnd())
{}

__host__ __device__
bool DP::VRPState::isAdmissible(OP::Variable::ValueType value) const
{
    return thrust::find(thrust::seq, admissibleValues.begin(), admissibleValues.end(), value) != admissibleValues.end();
}

__host__ __device__
bool DP::VRPState::isSelected(OP::Variable::ValueType value) const
{
    return thrust::find(thrust::seq, selectedValues.begin(), selectedValues.end(), value) != selectedValues.end();
}

__host__ __device__
DP::VRPState& DP::VRPState::operator=(VRPState const & other)
{
    cost = other.cost;
    selectedValues = other.selectedValues;
    admissibleValues = other.admissibleValues;
    return *this;
}

__host__ __device__
std::size_t DP::VRPState::sizeOfStorage(unsigned int variablesCount)
{
    return StaticVector<OP::Variable::ValueType>::sizeOfStorage(variablesCount) * 2;
}