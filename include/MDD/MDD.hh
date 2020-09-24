#pragma once

#include <vector>

#include <CustomTemplateLibrary/CTL.hh>
#include <MDD/Layer.hh>
#include <Problem/Variable.hh>

class MDD
{
    private:
        ctl::Vector<Layer> layers;

    public:
        static size_t getMemSize(std::vector<Variable> const & vars);

        MDD(uint maxWidth, std::vector<Variable> const & vars);

        void toGraphViz() const;

    private:
        void initialize();
};