include(ComponentsPackageFind)

set (SSL_INCLUDE_PATH "${PROJECT_SOURCE_DIR}/third-party")
set (SSL_LIB_PATH "${THIRD_PARTY_LIBS_PATH}/ssl")

# optional params should be set in a non-Brazil environment to assist the locating of this package
ComponentsPackageFind("SSL"                    # package name
                      "openssl/core.h"         # key header
                      "${SSL_FIND_COMPONENTS}" # library name
                      "${SSL_INCLUDE_PATH}"    # optional
                      "${SSL_LIB_PATH}"        # optional
                      ""                       # package comment prefix
                      )
