
#pragma once

#include <cstdint>
#include <vector>
#include <stack>
#include <set>
#include <mdd/node.hpp>
#include <mdd/edge.hpp>
#include <problem/problem.hpp>
#include <problem/variable.hpp>

class MDD {

private:
    int32_t num_variables;
    int32_t max_width;
    Problem *problem;
    std::vector< std::vector<Node> > nodes_per_layer;




public:
    MDD(int32_t vars, int32_t width, Problem problem);
    void build_mdd();
    std::vector< std::vector<Node> > get_nodes_per_layers() { return nodes_per_layer; }


    Node* expand_node( Node *node );
    void enforce_alldifff_sequential( Node *node );

    void add_edge(Node tail, Node head, std::vector<int32_t> edgeDomain );

    std::vector<Edge> maximize();
    std::vector<Edge> minimize();

    Problem* get_problem() { return problem; }


};
