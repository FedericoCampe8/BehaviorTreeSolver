

#pragma once

#include <vector>
#include "variable.hpp"
// #include "constraint_alldiff.hpp"
// #include "constraint_less_than.hpp"
#include "node.hpp"

#include "../system/system_export_defs.hpp"

class SYS_EXPORT_CLASS Problem {

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
