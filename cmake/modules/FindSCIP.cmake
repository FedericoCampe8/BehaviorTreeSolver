include(ComponentsPackageFind)

set (SCIP_INCLUDE_PATH "${PROJECT_SOURCE_DIR}/include/scip")
set (SCIP_LIB_PATH "${THIRD_PARTY_LIBS_PATH}/scip")

ComponentsPackageFind("SCIP"                    # package name
                      "soplex.h"                # key header
                      "${SCIP_FIND_COMPONENTS}" # library name
                      "${SCIP_INCLUDE_PATH}"    # optional
                      "${SCIP_LIB_PATH}"        # optional
                      ""                        # package comment prefix
                      )
