#include <algorithm>
#include <new>

#include <DP/AllDifferent.hh>
#include <MDD/MDD.hh>

size_t AllDifferent::State::getSizeStorage(const MDD *const mdd)
{
    return sizeof(int) * mdd->getLayersCount();
}

AllDifferent::State::State(State::Type type, std::size_t sizeStorage, std::byte * const storage) :
    DP::State(type, sizeStorage, storage),
    selectedValues(sizeStorage / sizeof(int), storage)
{}

AllDifferent::State & AllDifferent::State::operator=(AllDifferent::State const & other)
{
    assert(sizeStorage == other.sizeStorage);

    type = other.type;
    selectedValues = other.selectedValues;

    return *this;
}

bool AllDifferent::State::operator==(const AllDifferent::State &other)
{
    assert(type == other.type);

    return selectedValues == other.selectedValues;
}

bool AllDifferent::State::isValueSelected(int value) const
{
    return std::binary_search(selectedValues.begin(), selectedValues.end(), value);
}

void AllDifferent::State::addToSelectedValues(int value)
{
    selectedValues.emplaceBack(value);
    std::sort(selectedValues.begin(),selectedValues.end());
}

void AllDifferent::State::transitionFunction(int value, State * const child) const
{
    if (this->isValueSelected(value))
    {
        new (child) State(DP::State::Type::Impossible, child->sizeStorage, child->storage);
    }
    else
    {
        *child = *this;
        child->addToSelectedValues(value);
    }
}
