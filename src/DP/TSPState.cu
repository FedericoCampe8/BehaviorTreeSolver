#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "TSPState.cuh"

__host__ __device__
DP::TSPState::TSPState(OP::TSPProblem const * problem, std::byte* storage) :
    cost(INT_MAX),
    selectedValues(problem->vars.getCapacity(), storage),
    admissibleValues(problem->vars.getCapacity(), selectedValues.storageEnd())
{}

__host__ __device__
std::size_t DP::TSPState::sizeOfStorage(OP::TSPProblem const * problem)
{
    return StaticVector<unsigned int>::sizeOfStorage(problem->vars.getCapacity()) * 2;
}

__host__ __device__
bool DP::TSPState::isAdmissible(unsigned int value) const
{
    return thrust::binary_search(thrust::seq, admissibleValues.begin(), admissibleValues.end(), value);
}

__host__ __device__
bool DP::TSPState::isSelected(unsigned int value) const
{
    return thrust::find(thrust::seq, selectedValues.begin(), selectedValues.end(), value) != selectedValues.end();
}

__host__ __device__
void DP::TSPState::addToAdmissibles(unsigned int value)
{
    assert(not isAdmissible(value));

    admissibleValues.pushBack(value);
    thrust::sort(thrust::seq, admissibleValues.begin(),admissibleValues.end());
}

__host__ __device__
DP::TSPState& DP::TSPState::operator=(TSPState const & other)
{
    cost = other.cost;
    selectedValues = other.selectedValues;
    admissibleValues = other.admissibleValues;
    return *this;
}

__host__ __device__
void DP::TSPState::reset(TSPState& state)
{
    state.cost = INT_MAX;
    state.selectedValues.clear();
    state.admissibleValues.clear();
}