//
// Copyright OptiLab 2020. All rights reserved.
//
// All different constraint based on BT optimization.
//

#pragma once

#include <cstdint>  // for uint32_t
#include <limits>   // for std::numeric_limits
#include <vector>

#include <sparsepp/spp.h>


#include "system/system_export_defs.hpp"

namespace mdd {

class SYS_EXPORT_CLASS NodeDomain {

    public:
        NodeDomain( ) { }
        // NodeDomain( std::vector<int64_t>& values ) { pAvailableValues = values; }

        void addValue(int64_t value) { pAvailableValues.push_back(value); }
        bool removeValue(int64_t value);
        std::vector<int64_t>& getValues() { return pAvailableValues; }
        bool isValueInDomain( int64_t value ); 

        int getSize() { return pAvailableValues.size(); }  

        bool operator==(NodeDomain& other);


    private:
        std::vector<int64_t> pAvailableValues;
 
};


}  // namespace mdd
