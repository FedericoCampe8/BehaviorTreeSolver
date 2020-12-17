#pragma once

#include <Containers/RuntimeArray.cuh>
#include <Utils/Memory.cuh>

namespace Misc
{
    template<typename T>
    unsigned int getSizeOfRawArrayOfStates(unsigned int variablesCount, unsigned int size);

    template<typename T>
    T* getRawArrayOfStates(unsigned int variablesCount, unsigned int size, std::byte* memory);
}

template<typename T>
unsigned int Misc::getSizeOfRawArrayOfStates<T>(unsigned int variablesCount, unsigned int size)
{
    unsigned int stateSize = sizeof(T);
    unsigned int stateStorageSize = T::sizeOfStorage(variablesCount);
    return (stateSize * size)  + (stateStorageSize * size);
}

template<typename T>
T* Misc::getRawArrayOfStates<T>(unsigned int variablesCount, unsigned int size, std::byte* memory)
{
    unsigned int stateSize = sizeof(T);
    unsigned int stateStorageSize = T::sizeOfStorage(variablesCount);

    T* states = reinterpret_cast<T*>(memory);
    std::byte* statesStorages = &memory[stateSize * size];

    for(unsigned int stateIdx = 0; stateIdx < size; stateIdx += 1)
    {
        new (&states[stateIdx]) T(variablesCount, &statesStorages[stateStorageSize * stateIdx]);
    };

    return states;
}