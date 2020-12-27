#include <thrust/find.h>

#include "State.cuh"

__host__ __device__
DP::State::State(OP::Problem const * problem, std::byte* storage) :
    cost(0u),
    selectedValues(problem->variables.getCapacity(), reinterpret_cast<uint8_t*>(storage)),
    admissibleValues(problem->variables.getCapacity(), selectedValues.LightArray<uint8_t>::end())
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
void DP::State::removeFromAdmissibles(unsigned int value)
{
    uint8_t const * const admissibleValuesEnd = thrust::remove(thrust::seq, admissibleValues.begin(), admissibleValues.end(), static_cast<uint8_t>(value));
    if (admissibleValuesEnd != admissibleValues.end())
    {
        uint8_t* tmp = const_cast<uint8_t*>(admissibleValuesEnd);
        uint8_t* end = admissibleValues.end();
        unsigned int distance = tmp -  admissibleValues.end();
        unsigned int size = admissibleValues.indexOf(admissibleValuesEnd);
        admissibleValues.resize(size);
    }

    assert(not isAdmissible(value));
}


__host__ __device__
std::size_t DP::State::sizeOfStorage(OP::Problem const * problem)
{

    return LightArray<uint8_t>::sizeOfStorage(problem->variables.getCapacity()) * 2;
}
__host__ __device__
void DP::State::print() const
{
    printf("[DEBUG] Selected values: ");
    selectedValues.print(false);
    printf(" | Admissible values: ");
    admissibleValues.print();
}

