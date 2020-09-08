# ComponentsPackageFind - This is a general tool for anyone who wishes to use a package with a set
#                          of libraries and CMake
#
#
#     Arguments:
#
#       PACKAGE_NAME - Name of the package to look for (e.g. "Pryon" or "PryonOpenFST")
#
#       INDICATOR_HEADER_WITH_PREFIX - Header (or other) file to look for with path relative to
#                                      top level of package (or at least what would used for a
#                                      #include (e.g. "google/protobuf/descriptor.proto").
#
#       COMPONENTS - List of components (different libraries) desired.  This a CMake style list
#                    (string separated by ";").  This should generally just be fed the value of
#                    ${PACKAGE_NAME}_FIND_COMPONENTS that was passed into find_package() as
#                    the COMPONENTS parameter
#
#       INCLUDE_PATH - Extra path to include along with default for find_path().
#                      This is especially useful in an environment where the location of
#                      the headers can be specified with a separate variable.
#                      Can be empty (in that case, it doesn't really do anything).
#
#       LIB_PATH - Extra path to include along with default for find_library().
#                  This is especially useful in an environment where the location of the
#                  libraries can be specified with a separate variable.
#                  Can be empty (in that case, it doesn't really do anything).
#
#       MODULE_PREFIX - Package-specific prefix on component file names.
#
#
# Sets variables:
#
#   ${PACKAGE_NAME}_INCLUDE_DIR: Root of the include structure for this package
#                                (i.e. for "#include <a/b/file.h>", this directory contains
#                                 a/b/file.h)
#                                Use this result with include_directories().
#
#   ${PACKAGE_NAME}_${COMPONENT_NAME}_LIB: The full path (plus filename) /path/to/lib/COMPONENT.a or
#                                          /path/to/lib/COMPONENT.so.
#                                          Use this result with link_libraries() or
#                                          target_link_libraries().
#
#   ${PACKAGE_NAME}_LIBS: CMake-style list of the desired component libraries found (all of the
#                         ${PACKAGE_NAME}_${COMPONENT_NAME}_LIB values separated by ";")
#

include(PackageMacros)

function(ComponentsPackageFind
           PACKAGE_NAME
           INDICATOR_HEADER_WITH_PATH_PREFIX
           COMPONENTS
           INCLUDE_PATH
           LIB_PATH
           MODULE_PREFIX)

######## Input validation ########

# INCLUDE_PATH and LIB_PATH are allowed to be empty, so no validation needed for those
# Instead, checking to see if the default CMake path should be overridden

if(NOT PACKAGE_NAME)
  message(FATAL_ERROR "ERROR: In ComponentsPackageFind, PACKAGE_NAME input parameter is empty!")
endif(NOT PACKAGE_NAME)

if(NOT INDICATOR_HEADER_WITH_PATH_PREFIX)
  message(FATAL_ERROR "ERROR: In ComponentsPackageFind, ${PACKAGE_NAME} INDICATOR_HEADER_WITH_PATH_PREFIX input parameter is empty!")
endif(NOT INDICATOR_HEADER_WITH_PATH_PREFIX)

if(NOT COMPONENTS)
  message(FATAL_ERROR "ERROR: In ComponentsPackageFind, ${PACKAGE_NAME} COMPONENTS input parameter is empty!")
else()
  list(REMOVE_DUPLICATES COMPONENTS)
  list(SORT COMPONENTS)
endif(NOT COMPONENTS)

# This macro handles letting the user specify search-locations externally,
# i.e. on the cmake command-line
process_standard_input_variables(${PACKAGE_NAME})


######## Preliminary processing ########

# Helpful message to the operator indicating the package to search for
message(STATUS "Looking for ${PACKAGE_NAME}...")

######## INCLUDE Processing ########

# This if statement avoids repeating cached work
if(NOT ${PACKAGE_NAME}_INCLUDE_DIR)

  # This peels out the directory of the indicator-header, if it exists.
  #  (e.g. a/b/file.h becomes INDICATOR_FILE_DIR = "a/b")
  #
  # The INDICATOR_FILE_DIR is labeled as a suffix because it is considered so for CMake's find_path
  # (it is a suffix to be added to each filepath on the search path used by CMake to find the file)
  get_filename_component(INDICATOR_FILE_DIR ${INDICATOR_HEADER_WITH_PATH_PREFIX} DIRECTORY)

  # Search for the given indicator file and put the result in the ${PACKAGE_NAME}_INCLUDE_DIR var.
  # If found, the variable will contain the either value "/full/path/to", which is the directory
  # containing either "file.h" or "a/b/file.h" (where "a/b" is the INDICATOR_FILE_DIR determined above),
  # and if not found along the search path anywhere, the value will be "${PACKAGE_NAME}_INCLUDE_DIR-NOTFOUND")
  # Naturally, we only want /full/path/to if it contains a/b/file.h, so we have to handle such cases manually.

  list(APPEND PATH_SUFFIX "x86") # x86 is used by Android multi-target installations

  find_header_multipass() # it's a macro, defined in PackageMacros

endif()

# Ensure that the value of this variable is transmitted to a higher scope (since we're inside a
# function, but also want to be able to query this variable from CMakeLists.txt and elsewhere)
set(${PACKAGE_NAME}_INCLUDE_DIR "${${PACKAGE_NAME}_INCLUDE_DIR}" PARENT_SCOPE)

######## LIB Processing ########

# Check to see if the desired components can be found
message(STATUS "Looking for ${PACKAGE_NAME} components:")

set(${PACKAGE_NAME}_LIBS "")
foreach(COMP ${COMPONENTS})
    # Try to find the given component
    set(NAMING_OPTIONS "${MODULE_PREFIX}${COMP}" "${MODULE_PREFIX}${COMP}.a")
    find_library_multipass(${PACKAGE_NAME}_${COMP}_LIB "${NAMING_OPTIONS}")

    # Report status and, if found, add to library list
    if("${${PACKAGE_NAME}_${COMP}_LIB}" MATCHES "-NOTFOUND$")
        message(STATUS "  ${COMP} - MISSING")
    else()
        message(STATUS "  ${COMP}")
        set(${PACKAGE_NAME}_${COMP}_LIB "${${PACKAGE_NAME}_${COMP}_LIB}" PARENT_SCOPE)
        set(${CMAKE_FIND_PACKAGE_NAME}_${COMP}_FOUND True)
        list(APPEND ${PACKAGE_NAME}_LIBS "${${PACKAGE_NAME}_${COMP}_LIB}")
    endif()

endforeach()

# Ensure that the value of this variable is transmitted to a higher scope
set(${PACKAGE_NAME}_LIBS "${${PACKAGE_NAME}_LIBS}" PARENT_SCOPE)

######## Final processing ########

# These two lines use a standard part of CMake to provide the basic
# variable defines and messages expected from a Find___ module.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(${PACKAGE_NAME}
    REQUIRED_VARS "${PACKAGE_NAME}_INCLUDE_DIR" "${PACKAGE_NAME}_LIBS"
    HANDLE_COMPONENTS)

# Mark these as "advanced" variables, so they only show up when doing
# "advanced" CMake configuration
mark_as_advanced("${PACKAGE_NAME}_INCLUDE_DIR" "${PACKAGE_NAME}_LIBS")

# Generate imported targets
foreach(COMP ${COMPONENTS})
  if(${PACKAGE_NAME}_${COMP}_FOUND AND NOT TARGET ${PACKAGE_NAME}::${COMP})
    if(${PACKAGE_NAME}_${COMP}_LIB MATCHES "[.](so|dylib)$")
      add_library(${PACKAGE_NAME}::${COMP} SHARED IMPORTED)
    elseif(${PACKAGE_NAME}_${COMP}_LIB MATCHES "[.](o|a)$")
      add_library(${PACKAGE_NAME}::${COMP} STATIC IMPORTED)
    else() # e.g. .dll is used for both shared and static libs on Windows
      add_library(${PACKAGE_NAME}::${COMP} UNKNOWN IMPORTED)
    endif()
    set_target_properties(${PACKAGE_NAME}::${COMP} PROPERTIES
      IMPORTED_LOCATION "${${PACKAGE_NAME}_${COMP}_LIB}"
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      INTERFACE_INCLUDE_DIRECTORIES "${${PACKAGE_NAME}_INCLUDE_DIR}"
      )
  endif()
endforeach()

endfunction(ComponentsPackageFind)
