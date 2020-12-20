#include <thrust/find.h>

#include "State.cuh"


__host__ __device__
std::byte* DP::State::mallocStorages(unsigned int statesCount, OP::Problem const & problem, Memory::MallocType mallocType)
{
    return Memory::safeMalloc(sizeOfStorage(problem) * statesCount, mallocType);
}

__host__ __device__
std::size_t DP::State::sizeOfStorage(OP::Problem const & problem)
{
    return Vector<uint8_t>::sizeOfStorage(problem.variables.getCapacity()) * 2;
}

__host__ __device__
std::byte* DP::State::storageEnd() const
{
    return admissibleValues.storageEnd();
}

__host__ __device__
DP::State::State(OP::Problem const & problem, std::byte* storage) :
    cost(0u),
    selectedValues(problem.variables.getCapacity(), storage),
    admissibleValues(problem.variables.getCapacity(), selectedValues.storageEnd())
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