include(ComponentsPackageFind)

set (PROTOCOLBUFFER_INCLUDE_PATH "${PROJECT_SOURCE_DIR}/third-party/protobuf_3_12_2")
set (PROTOCOLBUFFER_LIB_PATH "${THIRD_PARTY_LIBS_PATH}/protobuf_3_12_2")

ComponentsPackageFind("ProtocolBuffer"                    # package name
                      "google/protobuf/descriptor.h"      # key header
                      "${ProtocolBuffer_FIND_COMPONENTS}" # library name
                      "${PROTOCOLBUFFER_INCLUDE_PATH}"    # optional
                      "${PROTOCOLBUFFER_LIB_PATH}"        # optional
                      ""                                  # package comment prefix
                      )
