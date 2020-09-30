#pragma once

#include <cstddef>

#include <CustomTemplateLibrary/CTL.hh>

class MDD;

namespace DP
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
            State(Type type, std::size_t sizeStorage, std::byte * const storage);
    };
}
