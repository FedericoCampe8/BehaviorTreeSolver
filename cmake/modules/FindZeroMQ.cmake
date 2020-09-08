include(SimplePackageFind)

set (ZEROMQ_INCLUDE_PATH "${PROJECT_SOURCE_DIR}/third-party")
set (ZEROMQ_LIB_PATH "${THIRD_PARTY_LIBS_PATH}/zeromq")

SimplePackageFind("ZeroMQ"                 # package name
                  "zeromq/zmq.h"           # key header
                  "zmq"                    # library name
                  "${ZEROMQ_INCLUDE_PATH}" # optional
                  "${ZEROMQ_LIB_PATH}"     # optional
                  )
