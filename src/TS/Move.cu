#include "Move.cuh"

__host__ __device__
TS::Move::Move(unsigned int fromVariable, ValueType fromValue, ValueType toValue) :
    fromVariable(fromVariable),
    fromValue(fromValue),
    toValue(toValue)
{}

__host__ __device__
bool TS::Move::operator==(TS::Move const & other)
{
    return
        fromVariable == other.fromVariable and
        fromValue == other.fromValue and
        toValue == other.toValue;
}