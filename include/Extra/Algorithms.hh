#pragma once

#ifdef GPU
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/equal.h>
#include <thrust/find.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/swap.h>
#endif

#include <algorithm>
#include <execution>
#include <iterator>
#include <utility>

#include <Extra/Utils/Platform.hh>

namespace Extra::Algorithms
{
    // Declarations
    template<typename T>
    __host__ __device__ int distance(T const * first, T const *  last);
    template<typename T>
    __host__ __device__ void sort(T * first, T * last);
    template<typename T>
    __host__ __device__ bool binary_search(T * first, T * last, T const & value);
    template<typename T>
    __host__ __device__ T * copy(T * first, T * last, T * d_first);
    template<typename T>
    __host__ __device__ T * find(T * first, T * last, T const & value);
    template<typename T>
    __host__ __device__ void swap(T & a, T & b);
    template<typename T>
    __host__ __device__ bool equal(T * first1, T * last1, T * first2);


    // Definitions
    template<typename T>
    __host__ __device__
    int distance(T * first,  T * last)
    {
#ifdef __CUDA_ARCH__
        return static_cast<int>(thrust::distance(first,last));
#else
        return static_cast<int>(std::distance(first,last));
#endif
    }

    template<typename T>
    __host__ __device__
    void sort(T * first, T * last)
    {
#ifdef __CUDA_ARCH__
        thrust::sort(thrust::seq,first,last);
#else
        std::sort(std::execution::seq,first,last);
#endif
    }

    template<typename T>
    __host__ __device__
    bool binary_search(T * first, T * last, T const & value)
    {
#ifdef __CUDA_ARCH__
         return thrust::binary_search(thrust::seq,first,last,value);
#else
         return std::binary_search(first,last,value);
#endif
    }

    template<typename T>
    __host__ __device__
    T * copy(T * first, T * last, T * d_first)
    {
#ifdef __CUDA_ARCH__
        return thrust::copy(thrust::seq,first,last,d_first);
#else
        return std::copy(std::execution::seq,first,last,d_first);
#endif
    }

    template<typename T>
    __host__ __device__
    T * find(T * first, T * last, T const & value)
    {
#ifdef __CUDA_ARCH__
        return thrust::find(thrust::seq,first,last,value);
#else
        return std::find(std::execution::seq,first,last,value);
#endif
    }

    template<typename T>
    __host__ __device__
    void swap(T & a, T & b)
    {
#ifdef __CUDA_ARCH__
        thrust::swap(a,b);
#else
        std::swap(a,b);
#endif
    }

    template<typename T>
    __host__ __device__
    bool equal(T * first1, T * last1, T * first2)
    {
#ifdef __CUDA_ARCH__
        return thrust::equal(thrust::seq,first1,last1,first2);
#else
        return std::equal(std::execution::seq,first1,last1,first2);
#endif
    }

    template<typename T, typename P>
    T * remove_if (T * first, T * last, P pred)
    {
#ifdef __CUDA_ARCH__
        return thrust::remove_if(thrust::seq,first,last1,pred);
#else
        return std::remove_if(std::execution::seq,first,last,pred);
#endif
    }
};