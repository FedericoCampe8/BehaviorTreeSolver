

#pragma once

#include <vector>
#include <problem/variable.hpp>

class Problem {

private:
    std::vector<Variable> variables;
    // std::vector<ConstraintAllDiff> constraints_all_diff;
    // std::vector<ConstraintLessThan> constraints_less_than;

public:
    void add_variable(Variable var);
    // void add_constraint(ConstraintAllDiff constraint) { constraints_all_diff.push_back( constraint ); }
    // void add_constraint(ConstraintLessThan constraint) { constraints_less_than.push_back( constraint ); }

    std::vector<Variable> get_variables();



};
