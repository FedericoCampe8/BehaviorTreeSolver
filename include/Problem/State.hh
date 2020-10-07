#pragma once

#include <cstddef>

#include <Extra/Utils.hh>

class MDD;

namespace Problem
{
    class State
    {
        public:
            enum Type: uint {Root, Regular, Impossible, Uninitialized};
            Type type;
        protected:
            std::size_t const sizeStorage;
            std::byte * const storage;

        public:
            __device__ State(Type type, std::size_t sizeStorage, std::byte * const storage);
    };
}
