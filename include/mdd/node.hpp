
#pragma once

#include <cstdint>
#include <vector>
#include <problem/variable.hpp>
#include <mdd/edge.hpp>

class Edge;

class  Node {

private:
    uint32_t layer;
    Variable* variable;
    float optimization_value;
    Edge* selected_edge;
    std::vector<Edge> inEdges;
    std::vector<Edge> outEdges;
    std::vector<int> available_values;


public:
    Node( Variable* variable, uint32_t layer);
    std::vector<int>* get_values();
    uint32_t get_layer();
    void add_in_edge( Edge* edge );
    void add_out_edge( Edge* edge );

    void remove_in_edge( int position );
    void remove_out_edge( int position );

    std::vector<Edge> get_out_edges();
    std::vector<Edge> get_in_edges();
    Variable* get_variable();

    void set_optimization_value( float opt_value );
    float get_optimization_value( );

    void set_selected_edge( Edge* edge );
    Edge* get_selected_edge(  );


};

