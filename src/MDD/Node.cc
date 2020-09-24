#include<MDD/Node.hh>

int Node::nextID = 0;

Node::Node() :
    ID(nextID++)
{
}