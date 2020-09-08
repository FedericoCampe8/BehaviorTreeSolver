include(SimplePackageFind)

set (SOPLEX_INCLUDE_PATH "${PROJECT_SOURCE_DIR}/include/soplex")
set (SOPLEX_LIB_PATH "${THIRD_PARTY_LIBS_PATH}/soplex")

SimplePackageFind("Soplex"                 # package name
                  "soplex/soplex.h"        # key header
                  "soplex"                 # library name
                  "${SOPLEX_INCLUDE_PATH}" # optional
                  "${SOPLEX_LIB_PATH}"     # optional
                  )
