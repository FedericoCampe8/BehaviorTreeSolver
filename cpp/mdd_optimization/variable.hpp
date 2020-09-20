//
// Copyright OptiLab 2020. All rights reserved.
//
// All different constraint based on BT optimization.
//

#pragma once

#include <cstdint>  // for int64_t
#include <vector>

#include "system/system_export_defs.hpp"

namespace mdd {

class SYS_EXPORT_CLASS Variable {
public:
    Variable(uint32_t id, uint32_t layer, const std::vector<int64_t>& availableValues);

    /// Return this variable's unique identifier
    uint32_t getId() const noexcept { return pId; }

    /// Returns this variable' domain
    const std::vector<int64_t>& getAvailableValues() const noexcept { return pAvailableValues; }

private:
    /// Variable unique identifier
    uint32_t pId;

    /// MDD Layer this variable is at
    uint32_t pLayerIndex;

    /// This variable's domain
    std::vector<int64_t> pAvailableValues;
};

}  // namespace mdd
