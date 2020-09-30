#pragma once

#include <cstddef>

#include <CustomTemplateLibrary/CTL.hh>
#include <DP/State.hh>

class MDD;

struct AllDifferent
{
    class State : public DP::State
    {
        private:
            ctl::StaticVector<int> selectedValues;

        public:
            static size_t getSizeStorage(MDD const * const mdd);

            State(Type type, std::size_t sizeStorage, std::byte * const storage);

            State & operator=(State const & other);
            bool operator==(State const & other);

            void next(int value, State * const child) const;

        private:
            bool isValueSelected(int value) const;
            void addToSelectedValues(int value);
    };

    static uint getOptimalLayerWidth(uint variablesCount);
};

