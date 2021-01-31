#pragma once

#include <cassert>
#include <cstdio>
#include <cstring>
#include <thrust/copy.h>
#include <thrust/swap.h>
#include <Utils/Memory.cuh>
#include <Utils/TypeAlias.h>

// https://stackoverflow.com/questions/47981/how-do-you-set-clear-and-toggle-a-single-bit

class BitSet
{
    // Members
    protected:
    u32 capacity;
    u32* storage;

    // Functions
    public:
    __host__ __device__ BitSet(u32 capacity, u32* storage);
    __host__ __device__ BitSet(u32 capacity, Memory::MallocType mallocType);
    __host__ __device__ ~BitSet();
    __host__ __device__ inline bool contains(u32 value) const;
    __host__ __device__ inline std::byte* endOfStorage() const;
    __host__ __device__ inline void erase(u32 value);
    __host__ __device__ inline u32 getCapacity() const;
    __host__ __device__ u32 getSize() const;
    __host__ __device__ inline void insert(u32 value);
    __host__ __device__ void merge(BitSet const & other);
    __host__ __device__ BitSet& operator=(BitSet const & other);
    __host__ __device__ void print(bool endLine = true) const;
    __host__ __device__ inline static u32 sizeOfStorage(u32 capacity);
    __host__ __device__ inline static void swap(BitSet& bs0, BitSet& bs1);
    private:
    __host__ __device__ inline static u32 chunksCount(u32 capacity);
    __host__ __device__ inline static u32 chunkIndex(u32 value);
    __host__ __device__ inline static u32 chunkOffset(u32 value);
    __host__ __device__ static u32* mallocStorage(u32 capacity, Memory::MallocType mallocType);
};

__host__ __device__
BitSet::BitSet(u32 capacity, u32* storage) :
    capacity(capacity),
    storage(storage)
{}

__host__ __device__
BitSet::BitSet(u32 capacity, Memory::MallocType mallocType) :
    BitSet(capacity, mallocStorage(capacity, mallocType))
{}

__host__ __device__
BitSet::~BitSet()
{
    free(storage);
}

__host__ __device__
bool BitSet::contains(u32 value) const
{
    assert(value < capacity);
    u32 const chunkIndex = BitSet::chunkIndex(value);
    u32 const chunkOffset = BitSet::chunkOffset(value);
    u32 const mask = 1u;
    return static_cast<bool>((storage[chunkIndex] >> chunkOffset) & mask);
}

__host__ __device__
std::byte* BitSet::endOfStorage() const
{
    u32 const chunksCount = BitSet::chunksCount(capacity);
    return reinterpret_cast<std::byte*>(storage + chunksCount);
}

__host__ __device__
void BitSet::erase(u32 value)
{
    assert(value < capacity);
    u32 const chunkIndex = BitSet::chunkIndex(value);
    u32 const chunkOffset = BitSet::chunkOffset(value);
    u32 const mask = ~(1u << chunkOffset);
    storage[chunkIndex] &= mask;
}

__host__ __device__
u32 BitSet::getCapacity() const
{
    return capacity;
}

__host__ __device__
u32 BitSet::getSize() const
{
    u32 size = 0;
    u32 const chunksCount = BitSet::chunksCount(capacity);
    for(u32 chunkIndex = 0; chunkIndex < chunksCount; chunkIndex += 1)
    {
#ifdef __CUDA_ARCH__
        size += __popc(storage[chunkIndex]);
#else
        size += __builtin__popcount(storage[chunkIndex])
#endif
    }
    return size;
}

__host__ __device__
void BitSet::insert(u32 value)
{
    assert(value < capacity);
    u32 const chunkIndex = BitSet::chunkIndex(value);
    u32 const chunkOffset = BitSet::chunkOffset(value);
    u32 const mask = 1u << chunkOffset;
    storage[chunkIndex] |= mask;
}

__host__ __device__
void BitSet::merge(BitSet const & other)
{
    assert(chunksCount(capacity) == other.chunksCount(other.capacity));
    u32 const chunksCount = BitSet::chunksCount(capacity);
    for(u32 chunkIndex = 0; chunkIndex < chunksCount; chunkIndex += 1)
    {
        storage[chunkIndex] |= other.storage[chunkIndex];
    }
}

__host__ __device__
BitSet& BitSet::operator=(BitSet const & other)
{
    assert(chunksCount(capacity) == other.chunksCount(other.capacity));
    memcpy(storage, other.storage, sizeOfStorage(other.capacity));
    return *this;
}

__host__ __device__
void BitSet::print(bool endLine) const
{
    printf("[");
    if (capacity > 0)
    {
        u32 value = 0;
        printf(contains(value) ? "1" : "0");
        for (value += 1; value < capacity; value += 1)
        {
            printf(",");
            printf(contains(value) ? "1" : "0");
        }
    }
    printf(endLine ? "]\n" : "]");
}

__host__ __device__
u32 BitSet::sizeOfStorage(u32 capacity)
{
    return sizeof(u32) * BitSet::chunksCount(capacity);
}

__host__ __device__
void BitSet::swap(BitSet& bs0, BitSet& bs1)
{
    thrust::swap(bs0.capacity, bs1.capacity);
    thrust::swap(bs0.storage, bs1.storage);
}

__host__ __device__
u32 BitSet::chunksCount(u32 capacity)
{
    return (capacity + 31) / 32;
}

__host__ __device__
u32 BitSet::chunkIndex(u32 value)
{
    return value / 32;
}

__host__ __device__
u32 BitSet::chunkOffset(u32 value)
{
    return value % 32;
}

__host__ __device__
u32* BitSet::mallocStorage(u32 capacity, Memory::MallocType mallocType)
{
    u32 const storageSize = sizeOfStorage(capacity);
    return reinterpret_cast<u32*>(Memory::safeMalloc(storageSize, mallocType));
}