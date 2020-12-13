#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "TSPState.cuh"

__host__ __device__
DP::TSPState::TSPState(OP::TSPProblem const * problem, std::byte* storage) :
    active(false),
    exact(true),
    cost(UINT32_MAX),
    selectedValues(problem->vars.size, storage),
    admissibleValues(problem->vars.size, selectedValues.getStorageEnd())
{}

__host__ __device__
std::size_t DP::TSPState::sizeOfStorage(OP::TSPProblem const * problem)
{
    return StaticVector<int32_t>::sizeofStorage(problem->vars.size) * 2;
}

__host__ __device__
bool DP::TSPState::isAdmissible(int value) const
{
    return thrust::binary_search(thrust::seq, admissibleValues.begin(), admissibleValues.end(), value);
}

__host__ __device__
bool DP::TSPState::isSelected(int value) const
{
    return thrust::find(thrust::seq, selectedValues.begin(), selectedValues.end(), value) != selectedValues.end();
}

__host__ __device__
void DP::TSPState::addToAdmissibles(int value)
{
    assert(not isAdmissible(value));

    admissibleValues.pushBack(value);
    thrust::sort(thrust::seq, admissibleValues.begin(),admissibleValues.end());
}

__host__ __device__
DP::TSPState& DP::TSPState::operator=(TSPState const & other)
{
    active = other.active;
    exact = other.exact;
    cost = other.cost;
    selectedValues = other.selectedValues;
    admissibleValues = other.admissibleValues;
    return *this;
}

__host__ __device__
void DP::TSPState::reset(TSPState& state)
{
    state.active = false;
    state.exact = true;
    state.cost = UINT32_MAX;
    state.selectedValues.clear();
    state.admissibleValues.clear();
}

__host__ __device__
void DP::TSPState::print()
{
    auto bool2Str = [&] (bool const & b) -> char const *
    {
        return b ? "T" : "F";
    };

    printf("Active: %s | Exact: %s | Cost: %d | Details: ", bool2Str(active) , bool2Str(exact), cost);
    selectedValues.print(false);
    admissibleValues.print();
}
