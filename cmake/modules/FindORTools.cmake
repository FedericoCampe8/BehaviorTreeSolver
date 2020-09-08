include(ComponentsPackageFind)

set (ORTOOLS_INCLUDE_PATH "${PROJECT_SOURCE_DIR}/include/or-tools")
set (ORTOOLS_LIB_PATH "${THIRD_PARTY_LIBS_PATH}/or-tools")

ComponentsPackageFind("ORTools"                    # package name
                      "ortools/constraint_solver/constraint_solver.h" # key header
                      "${ORTools_FIND_COMPONENTS}" # library name
                      "${ORTOOLS_INCLUDE_PATH}"    # optional
                      "${ORTOOLS_LIB_PATH}"        # optional
                      ""                           # package comment prefix
                      )
