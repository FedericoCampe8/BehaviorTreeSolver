#include <problem/variable.hpp>

Variable::Variable( uint32_t id, uint32_t layer, std::vector<int> available_values )
{
    this->id = id;
    this->layer_index = layer;
    this->available_values = available_values;
}

uint32_t Variable::get_id() { return id; }
std::vector<int> Variable::get_available_values() { return available_values; }
