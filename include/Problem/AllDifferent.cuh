#pragma once

#include <cstdint>

#include "Extra/Containers/StaticVector.cuh"
#include "Problem/DPModel.cuh"

class AllDifferent : public DPModel
{
    public:
        class State : public DPModel::State
        {
            // Overload warning suppression
            using DPModel::State::operator=;
            using DPModel::State::operator==;

            private:
                StaticVector<int> selectedValues;

            public:
                __device__ State(std::size_t storageMemSize, std::byte* storageMem);
                __device__ State & operator=(State const & other);
                __device__ bool operator==(State const & other) const;
                __device__ unsigned int getSimilarity(State const & other) const;
                __device__ static void makeRoot(State* state);
                __device__ static std::size_t sizeofStorage(unsigned int i);
                __device__ static void getNextStates(State const * currentState, int minValue, int maxValue, State* nextStates);
            private:
                __device__ bool isSelected(int value) const;
                __device__ void addToSelected(int value);

                friend class AllDifferent;
        };
};

