#include <thrust/copy.h>
#include <thrust/find.h>
#include <thrust/remove.h>

#include "State.cuh"

__host__ __device__
DP::State::State(OP::Problem const * problem, std::byte* storage) :
    cost(MaxCost),
    selectedValues(problem->variables.getCapacity(), reinterpret_cast<OP::ValueType*>(storage)),
    admissibleValues(problem->variables.getCapacity(), selectedValues.LightArray<OP::ValueType>::end())
{}

__host__ __device__
bool DP::State::isAdmissible(OP::ValueType value) const
{
    return thrust::find(thrust::seq, admissibleValues.begin(), admissibleValues.end(), value) != admissibleValues.end();
}

__host__ __device__
std::byte* DP::State::mallocStorages(OP::Problem const * problem, unsigned int statesCount, Memory::MallocType mallocType)
{
    return Memory::safeMalloc(sizeOfStorage(problem) * statesCount, mallocType);
}

__host__ __device__
void DP::State::operator=(DP::State const & other)
{
    cost = other.cost;
    selectedValues.resize(other.selectedValues.getSize());
    thrust::copy(thrust::seq, other.selectedValues.begin(), other.selectedValues.end(), selectedValues.begin());
    admissibleValues.resize(other.admissibleValues.getSize());
    thrust::copy(thrust::seq, other.admissibleValues.begin(), other.admissibleValues.end(), admissibleValues.begin());
}

__host__ __device__
void DP::State::removeFromAdmissibles(OP::ValueType value)
{
    OP::ValueType const * const admissibleValuesEnd = thrust::remove(thrust::seq, admissibleValues.begin(), admissibleValues.end(), value);
    if (admissibleValuesEnd != admissibleValues.end())
    {
        unsigned int const size = admissibleValues.indexOf(admissibleValuesEnd);
        admissibleValues.resize(size);
    }
}

__host__ __device__
unsigned int DP::State::sizeOfStorage(OP::Problem const * problem)
{
    return
        LightArray<OP::ValueType>::sizeOfStorage(problem->variables.getCapacity()) + // selectedValues
        LightArray<OP::ValueType>::sizeOfStorage(problem->variables.getCapacity()) + // admissibleValues
        8 * 2; // Alignment
}

__host__ __device__
void DP::State::reset()
{
    cost = MaxCost;
    selectedValues.clear();
    admissibleValues.clear();
}