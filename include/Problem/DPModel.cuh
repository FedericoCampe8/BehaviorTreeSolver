#pragma once

#include <cstddef>
#include <cstdint>

class DPModel
{
    public:
        class State
        {
            public:
                enum Type : uint16_t {Regular, Impossible, Uninitialized};
                Type type;

            public:
                __device__ inline State();
                __device__ inline virtual State& operator=(State const & other);
                __device__ inline virtual bool operator==(State const & other) const;
                __device__ inline static void makeRoot(State* s);
        };
};

__device__
DPModel::State::State() :
    type(Type::Uninitialized)
{}

__device__
DPModel::State & DPModel::State::operator=(State const & other)
{
    type = other.type;
    return *this;
}

__device__
bool DPModel::State::operator==(State const & other) const
{
    return type == other.type;
}

__device__
void DPModel::State::makeRoot(State* state)
{
    state->type = Type::Regular;
}