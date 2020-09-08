include(ComponentsPackageFind)

set (BOOST_INCLUDE_PATH "${PROJECT_SOURCE_DIR}/third-party/boost")
set (BOOST_LIB_PATH "${THIRD_PARTY_LIBS_PATH}/boost")

# optional params should be set in a non-Brazil environment to assist the locating of this package
ComponentsPackageFind("Boost"                    # package name
                      "asio.hpp"                 # key header
                      "${Boost_FIND_COMPONENTS}" # library name
                      "${BOOST_INCLUDE_PATH}"    # optional
                      "${BOOST_LIB_PATH}"        # optional
                      ""                         # package comment prefix
                      )
