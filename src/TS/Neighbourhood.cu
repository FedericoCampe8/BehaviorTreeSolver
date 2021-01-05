#include <thrust/fill.h>
#include <Containers/LightVector.cuh>

#include "Attributes.cuh"
#include "Neighbourhood.cuh"

TS::Neighbourhood::Neighbourhood(OP::Problem const * problem, unsigned int tabuLength, Memory::MallocType mallocType)  :
    tabuLength(tabuLength),
    timestamp(0),
    valuesCount(problem->variables.getCapacity()),
    attributes(problem->variables.getCapacity() * valuesCount, mallocType)
{
    for(TS::Attributes* attribute = attributes.begin(); attribute != attributes.end(); attribute += 1)
    {
        new (attribute) TS::Attributes();
    }
}

__host__ __device__
TS::Attributes* TS::Neighbourhood::getAttributes(unsigned int variableIdx, unsigned int value) const
{
    return attributes[valuesCount * variableIdx + value];
}


void TS::Neighbourhood::operator=(Neighbourhood const & other)
{
    tabuLength = other.tabuLength;
    timestamp = other.timestamp;
    valuesCount = other.valuesCount;
    attributes = other.attributes;
}

void TS::Neighbourhood::update(LightVector<OP::Variable::ValueType> const * solution)
{
    for(unsigned int variableIdx = 0; variableIdx < solution->getSize(); variableIdx += 1)
    {
        OP::Variable::ValueType const value = *solution->at(variableIdx);
        attributes[valuesCount * variableIdx + value]->lastTimeSeen = timestamp;
    }

    timestamp += 1;
}

void TS::Neighbourhood::reset()
{
    for(TS::Attributes* attribute = attributes.begin(); attribute != attributes.end(); attribute += 1)
    {
        new (attribute) TS::Attributes();
    }
}
