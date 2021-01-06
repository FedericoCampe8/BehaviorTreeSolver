#include "../OP/Variable.cuh"

namespace TS
{
    class Move
    {
        // Aliases, Enums, ...
        public:
            using ValueType = OP::Variable::ValueType;

        // Members
        public:
            unsigned int fromVariable;
            ValueType fromValue;
            ValueType toValue;

        // Functions
        public:
            __host__ __device__ Move(unsigned int fromVariable, ValueType fromValue, ValueType toValue);
            __host__ __device__ bool operator==(Move const & other);
            __host__ __device__ void print(bool endLine = true) const;
    };

}