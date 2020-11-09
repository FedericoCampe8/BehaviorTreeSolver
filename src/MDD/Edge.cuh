#pragma once

#include <cstdint>

namespace MDD
{
    class Edge
    {
        public:
            enum Status {Active, NotActive};
            Status status: 1;
            uint16_t to: 15;

        public:
            __device__ Edge();
            __device__ Edge(unsigned int to);
            __device__ static void reset(Edge & edge);
            __device__ static bool isActive(Edge const & edge);
    };
}