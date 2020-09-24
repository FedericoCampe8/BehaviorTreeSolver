#pragma once

#include <cstddef>

#include <CustomTemplateLibrary/CTL.hh>

class AllDifferent
{
    class State
    {
        private:
            bool valid;
            ctl::Vector<int> selectedValues;

        public:
            static size_t getMemSize(uint varsCount);

            State();
            State(uint varsCount);
            State(State const * other);

            size_t getMemSize() const;

            void addValue(int value);
            bool containsValue(int value) const;
    };

    static State const * transitionFunction (State const * parent, int value);
};

