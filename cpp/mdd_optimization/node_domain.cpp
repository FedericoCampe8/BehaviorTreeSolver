#include "mdd_optimization/node_domain.hpp"


namespace mdd {

bool NodeDomain::removeValue(int64_t value)
{
    auto iter = std::find(pAvailableValues.begin(), pAvailableValues.end(), value);
    if (iter != pAvailableValues.end()) {
        pAvailableValues.erase( iter );
        return true;
    }
    return false;
}

bool NodeDomain::isValueInDomain(int64_t value)
{
    auto iter = std::find(pAvailableValues.begin(), pAvailableValues.end(), value);
    return iter != pAvailableValues.end();
}


bool NodeDomain::operator==(NodeDomain& other)
{
    if (pAvailableValues.size() != other.getSize()) {
        return false;
    }

    //TODO make sure they are sorted to begin with so I don't have to sort on each comparison
    std::sort( pAvailableValues.begin(), pAvailableValues.end()  );
    std::sort( other.getValues().begin(), other.getValues().end()  );

    for (int i = 0; i < pAvailableValues.size(); i++) {
        if ( pAvailableValues[i] != other.getValues()[i]) {
            return false;
        }
    }
    return true;
} 

}