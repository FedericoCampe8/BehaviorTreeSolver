
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "../system/system_export_defs.hpp"

class SYS_EXPORT_CLASS Variable {

private:
    uint32_t id;
    uint32_t layer_index;
    std::vector<int> available_values;

public:
    Variable( uint32_t id, uint32_t layer, std::vector<int> available_values );

    uint32_t get_id();
    std::vector<int> get_available_values();

};
