#pragma once

#include <cstdint>

class Edge
{
    public:
        enum Status {NotInitialized, Active, NotActive};
        uint16_t from;
        uint16_t to;
        int32_t value : 30;
    private:
        Status status : 2;

    public:
        __device__ Edge();
        __device__ Edge(unsigned int from, unsigned int to, int value);
        __device__ bool isActive() const;
        __device__ void deactivate();
};