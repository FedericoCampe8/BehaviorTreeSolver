#include "edge.hpp"

Edge::Edge(Node *tail, Node *head, int value)
{
    this->tail = tail;
    this->head = head;
    this->value = value;
}

void Edge::set_head( Node node )
{
    head = &node;
}

Node* Edge::get_head()
{
    return this->head;
}

Node* Edge::get_tail()
{
    return this->tail;
}

int Edge::get_value()
{
    return this->value;
}