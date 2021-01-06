#include "Neighbourhood.cuh"

TS::Neighbourhood::Neighbourhood(OP::Problem const * problem, unsigned int tabuListSize, Memory::MallocType mallocType)  :
    bestAvgCost(UINT_MAX),
    updatesCount(0),
    shortTermTabuMoves(tabuListSize * problem->variables.getCapacity(), mallocType),
    midTermTabuMoves(tabuListSize * problem->variables.getCapacity(), mallocType),
    longTermTabuMoves(tabuListSize * problem->variables.getCapacity(), mallocType)
{}

__host__ __device__
bool TS::Neighbourhood::isTabu(Move const * move) const
{
    for(unsigned int moveIdx = 0; moveIdx < shortTermTabuMoves.getSize(); moveIdx += 1)
    {
        if(*shortTermTabuMoves.at(moveIdx) == *move)
        {
            return true;
        }
    }
    for(unsigned int moveIdx = 0; moveIdx < midTermTabuMoves.getSize(); moveIdx += 1)
    {
        if(*midTermTabuMoves.at(moveIdx) == *move)
        {
            return true;
        }
    }
    for(unsigned int moveIdx = 0; moveIdx < longTermTabuMoves.getSize(); moveIdx += 1)
    {
        if(*longTermTabuMoves.at(moveIdx) == *move)
        {
            return true;
        }
    }
    return false;
};


void TS::Neighbourhood::update(LightVector<ValueType> const * solution)
{

    for (unsigned int variableIdx = 0; variableIdx < solution->getSize() - 1; variableIdx += 1)
    {
        unsigned int fromVariable = variableIdx;
        ValueType fromValue = *solution->at(variableIdx);
        ValueType toValue = *solution->at(variableIdx + 1);
        Move move(fromVariable, fromValue, toValue);

        if(rand() % 100 < 100)
        {

            if (updatesCount % 1 == 0)
            {
                if (shortTermTabuMoves.isFull())
                {
                    shortTermTabuMoves.popFront();
                }
                shortTermTabuMoves.pushBack(&move);
            }

            if (updatesCount % 10 == 0)
            {
                if (midTermTabuMoves.isFull())
                {
                    midTermTabuMoves.popFront();
                }
                midTermTabuMoves.pushBack(&move);
            }

            if (updatesCount % 100 == 0)
            {
                if (longTermTabuMoves.isFull())
                {
                    longTermTabuMoves.popFront();
                }
                longTermTabuMoves.pushBack(&move);
            }
        }
    }

    updatesCount += 1;
}