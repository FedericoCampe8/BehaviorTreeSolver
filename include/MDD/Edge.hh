#pragma once

#include <Extra/Utils.hh>

class Edge
{
    public:
        enum Status {Valid, Invalid};
        Status status;
        uint to;
        int  value;

    public:
        __device__ Edge(uint to, int value);
        __device__ static bool isNotValid(Edge const & edge);
};
