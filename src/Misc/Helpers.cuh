#pragma once

#include <Containers/RuntimeArray.cuh>
#include <Utils/Memory.cuh>

namespace Misc
{
    template<typename T>
    T* getState(unsigned int variablesCount, Memory::MallocType mallocType);

    template<typename T>
    inline unsigned int getSizeOfRawArrayOfStates(unsigned int variablesCount, unsigned int size);

    template<typename T>
    T* getRawArrayOfStates(unsigned int variablesCount, unsigned int size, std::byte* memory);
}

template<typename T>
T* Misc::getState<T>(unsigned int variablesCount, Memory::MallocType mallocType)
{
    unsigned int const stateSize = sizeof(T);
    unsigned int const stateStorageSize = T::sizeOfStorage(variablesCount);
    std::byte* const memory = Memory::safeMalloc(stateSize + stateStorageSize, mallocType);
    T* const state = reinterpret_cast<T*>(memory);
    std::byte* const stateStorage = &memory[stateSize];
    new (state) T(variablesCount, stateStorage);

    return state;
}

template<typename T>
unsigned int Misc::getSizeOfRawArrayOfStates<T>(unsigned int variablesCount, unsigned int size)
{
    unsigned int const stateSize = sizeof(T);
    unsigned int const stateStorageSize = T::sizeOfStorage(variablesCount);
    return (stateSize * size)  + (stateStorageSize * size);
}

template<typename T>
T* Misc::getRawArrayOfStates<T>(unsigned int variablesCount, unsigned int size, std::byte* memory)
{
    unsigned int const stateSize = sizeof(T);
    unsigned int const stateStorageSize = T::sizeOfStorage(variablesCount);

    T* const states = reinterpret_cast<T*>(memory);
    std::byte* const statesStorages = &memory[stateSize * size];

    for(unsigned int stateIdx = 0; stateIdx < size; stateIdx += 1)
    {
        new (&states[stateIdx]) T(variablesCount, &statesStorages[stateStorageSize * stateIdx]);
    };

    return states;
}