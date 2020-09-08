include(ComponentsPackageFind)

if( UNIX AND NOT APPLE )
  set (GECODE_INCLUDE_PATH "${PROJECT_SOURCE_DIR}/include/gecode_linux")
elseif( APPLE )
  set (GECODE_INCLUDE_PATH "${PROJECT_SOURCE_DIR}/include/gecode_mac")
endif()

set (GECODE_LIB_PATH "${THIRD_PARTY_LIBS_PATH}/gecode")

# optional params should be set in a non-Brazil environment to assist the locating of this package
ComponentsPackageFind("Gecode"                    # package name
                      "gecode/driver.hh"          # key header
                      "${Gecode_FIND_COMPONENTS}" # library name
                      "${GECODE_INCLUDE_PATH}"    # optional
                      "${GECODE_LIB_PATH}"        # optional
                      ""                          # package comment prefix
                      )
