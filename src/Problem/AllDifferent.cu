#include <new>

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/logical.h>

#include <MDD/MDD.cuh>
#include <Problem/AllDifferent.cuh>

__device__
AllDifferent::State::State(std::size_t storageMemSize, std::byte * storageMem) :
    DPModel::State(),
    selectedValues(storageMemSize / sizeof(int), reinterpret_cast<int*>(storageMem))
{}

__device__
AllDifferent::State & AllDifferent::State::operator=(State const & other)
{
    DPModel::State::operator=(other);
    selectedValues = other.selectedValues;
    return *this;
}

__device__
bool AllDifferent::State::operator==(State const & other) const
{
    if(not DPModel::State::operator==(other))
    {
        return false;
    }

    assert(selectedValues.getSize() == other.selectedValues.getSize());
    auto isSelected = [&] (auto & v) -> bool
    {
        return  this->isSelected(v);
    };
    return thrust::all_of(thrust::seq, other.selectedValues.begin(), other.selectedValues.end(), isSelected);
}

__device__
void AllDifferent::State::makeRoot(State* state)
{
    DPModel::State::makeRoot(state);
}

__device__
void AllDifferent::State::getNextStates(State const * currentState, int minValue, int maxValue, State* nextStates)
{
    for(int i = 0; i < maxValue - minValue + 1; i += 1)
    {
        int value = minValue + i;
        State * nextState = &nextStates[i];

        if (currentState->isSelected(value))
        {
            nextState->type = DPModel::State::Impossible;
        }
        else
        {
            *nextState = *currentState;
            nextState->addToSelected(value);
        }
    }
}

__device__
bool AllDifferent::State::isSelected(int value) const
{
    return thrust::find(thrust::seq, selectedValues.begin(), selectedValues.end(), value) != selectedValues.end();
}

__device__
void AllDifferent::State::addToSelected(int value)
{
    selectedValues.pushBack(value);
}

__device__
std::size_t AllDifferent::State::sizeofStorage(unsigned int i)
{
    return StaticVector<int>::sizeofStorage(i);
}

__device__
unsigned int AllDifferent::State::getSimilarity(State const & other) const
{
    assert(selectedValues.getSize() == other.selectedValues.getSize());
    //Todo: Parallelize
    return thrust::inner_product(
        thrust::seq,
        selectedValues.begin(),
        selectedValues.end(),
        other.selectedValues.begin(),
        0,
        thrust::equal_to<int>(),
        thrust::plus<unsigned int>());
}
