#pragma once

#include <utility>
#include <thrust/swap.h>

#include <iterator>
#include <thrust/distance.h>

/*
 * References
 * - Sort: https://www.geeksforgeeks.org/in-place-merge-sort/
 * - Binary search: https://www.geeksforgeeks.org/binary-search/
 * - Copy: https://en.cppreference.com/w/cpp/algorithm/copy
 * - Find: https://en.cppreference.com/w/cpp/algorithm/find
 * - Equal: https://en.cppreference.com/w/cpp/algorithm/equal
 * - Remove if: https://en.cppreference.com/w/cpp/algorithm/remove
 */

namespace Algorithms
{
    // Declarations
    template<typename T>
    __host__ __device__ int distance(T * first, T *  last);
    template<typename T>
    __host__ __device__ void sort(T * first, T * last);
    template<typename T>
    __host__ __device__ void mergeSort(T * array, int l, int r);
    template<typename T>
    __host__ __device__ void merge(T * array, int start, int mid, int end);
    template<typename T>
    __host__ __device__ bool binary_search(T * first, T * last, T const & value);
    template<typename T>
    __host__ __device__ bool binarySearch(T * array, int l, int r, T const & x);
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
        int d = Algorithms::distance(first,last);
        if (d > 0)
        {
            mergeSort(first, 0, static_cast<unsigned int>(d - 1));
        }
    }

    template<typename T>
    __host__ __device__
    void mergeSort(T * array, int l, int r)
    {
        if (l < r)
        {
            // Same as (l + r) / 2, but avoids overflow for large l and r
            int m = l + (r - l) / 2;

            // Sort first and second halves
            mergeSort(array, l, m);
            mergeSort(array, m + 1, r);

            merge(array, l, m, r);
        }
    }

    template<typename T>
    __host__ __device__
    void merge(T * array, int start, int mid, int end)
    {
        int start2 = mid + 1;

        // If the direct merge is already sorted
        if (array[mid] <= array[start2])
        {
            return;
        }

        // Two pointers to maintain start of both arrays to merge
        while (start <= mid and start2 <= end)
        {
            // If element 1 is in right place
            if (array[start] <= array[start2])
            {
                start++;
            }
            else
            {
                T value = array[start2];
                int index = start2;

                // Shift all the elements between element 1 and element 2, right by 1.
                while (index != start)
                {
                    array[index] = array[index - 1];
                    index--;
                }
                array[start] = value;

                // Update all the pointers
                start++;
                mid++;
                start2++;
            }
        }
    }

    template<typename T>
    __host__ __device__
    bool binary_search(T * first, T * last,  T const & value)
    {
        int d = Algorithms::distance(first,last);
        return d > 0 ? Algorithms::binarySearch(first, 0, d - 1, value) : false;
    }

    template<typename T>
    __host__ __device__
    bool binarySearch(T * array, int l, int r, T const & x)
    {
        if (r >= l)
        {
            int mid = l + (r - l) / 2;

            // If the element is present at the middle itself
            if (array[mid] == x)
            {
                return true;
            }

            // If element is smaller than mid, then it can only be present in left subarray
            if (array[mid] > x)
            {
                return binarySearch(array, l, mid - 1, x);
            }

            // Else the element can only be present in right subarray
            return binarySearch(array, mid + 1, r, x);
        }

        // We reach here when element is not present in array
        return false;
    }

    template<typename T>
    __host__ __device__
    T * copy(T * first, T * last, T * d_first)
    {
        while (first != last)
        {
            *d_first++ = *first++;
        }
        return d_first;
    }

    template<typename T>
    __host__ __device__
    T * find(T * first, T * last, T const & value)
    {
        for (; first != last; ++first)
        {
            if (*first == value)
            {
                return first;
            }
        }
        return last;
    }

    template<typename T>
    __host__ __device__
    void swap(T * & a, T * & b)
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
        for (; first1 != last1; ++first1, ++first2)
        {
            if (!(*first1 == *first2))
            {
                return false;
            }
        }
        return true;
    }

    template<typename T, typename P>
    T * remove_if (T * first, T * last, P pred)
    {
        T * result = first;
        while (first != last)
        {
            if (not pred(*first))
            {
                *result = *first;
                ++result;
            }
            ++first;
        }
        return result;
    }
};