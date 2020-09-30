#include <DP/State.hh>

DP::State::State(Type type, std::size_t sizeStorage, std::byte * const storage) :
    type(type == Type::Root ? Type::Regular : type),
    sizeStorage(sizeStorage),
    storage(storage)
{
}

