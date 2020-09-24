#include <algorithm>
#include <new>

#include <DP/AllDifferent.hh>

size_t AllDifferent::State::getMemSize(uint varsCount)
{
    return sizeof(State) + ctl::Vector<int>::getMemSize(varsCount);
}

AllDifferent::State::State() :
    valid(false),
    selectedValues(0u)
{
}

AllDifferent::State::State(uint varsCount) :
    valid(true),
    selectedValues(varsCount)
{
}

AllDifferent::State::State(State const * other) :
    valid(true),
    selectedValues(&other->selectedValues)
{
}

size_t AllDifferent::State::getMemSize() const
{
    return sizeof(State) + selectedValues.getMemSize();
}

void AllDifferent::State::addValue(int value)
{
    assert(!containsValue(value));

    selectedValues.pushBack(value);
    std::sort(selectedValues.begin(),selectedValues.end());

}

bool AllDifferent::State::containsValue(int value) const
{
    return std::binary_search(selectedValues.begin(), selectedValues.end(), value);
}

AllDifferent::State const * AllDifferent::transitionFunction(AllDifferent::State const * parent, int value)
{
    AllDifferent::State* child = static_cast<AllDifferent::State*>(malloc(parent->getMemSize()));

    if (parent->containsValue(value))
    {
        new (child) State();
    }
    else
    {
        new (child) State(parent);
        child->addValue(value);
    }

    return child;
}
