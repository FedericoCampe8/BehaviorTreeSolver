
#pragma once

#include <cstdint>
#include <vector>

class Node;

class Edge {

private:
    Node* tail;
    Node* head;
    int value;



public:
    Edge( Node *tail, Node *head, int value );
    
    Node* get_head();
    Node* get_tail();
    int get_value();

    void set_head( Node* node );

};

