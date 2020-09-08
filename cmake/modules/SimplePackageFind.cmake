# SimplePackageFind - This is a general tool for anyone who wishes to use a package with a single
#                     library and CMake
#
#
#     Arguments:
#
#       PACKAGE_NAME - Name of the package to look for (e.g. "Gecode"
#
#       INDICATOR_HEADER_WITH_PREFIX - Header (or other) file to look for with path relative to
#                                      top level of package (or at least what would used for a
#                                      #include (e.g. "google/protobuf/descriptor.proto").
#
#       INCLUDE_PATH - Extra path to include along with default for find_path().
#                      This is especially useful when the location of
#                      the headers can be specified with a separate variable.
#                      Can be empty (in that case, it doesn't really do anything).
#
#       LIBRARY_SEARCH_NAME - Library name given to find_library function. Go to find_library
#                             function documentation for details. Can be empty
#                             (for, for instance, header-only libraries)
#
#       LIB_PATH - Extra path to include along with default for find_library().
#                  This is especially useful when the location of the
#                  libraries can be specified with a separate variable.
#                  Can be empty (in that case, it doesn't really do anything).
#
#
# Sets variables:
#
#   ${PACKAGE_NAME}_INCLUDE_DIR: Root of the include structure for this package
#                                (i.e. for "#include <a/b/file.h>", this directory contains
#                                 a/b/file.h)
#                                Use this result with include_directories().
#
#   ${PACKAGE_NAME}_LIB: Path to the library for this package (including the filename)
#

include(PackageMacros)

function(SimplePackageFind
           PACKAGE_NAME
           INDICATOR_HEADER_WITH_PATH_PREFIX
           LIBRARY_SEARCH_NAME
           INCLUDE_PATH
           LIB_PATH)

######## Input validation and processing ########

# INCLUDE_PATH and LIB_PATH are allowed to be empty, so no validation needed for those
# Instead, checking to see if the default CMake path should be overridden

if(NOT PACKAGE_NAME)
  message(FATAL_ERROR "ERROR: In SimplePackageFind, PACKAGE_NAME input parameter is empty!")
endif(NOT PACKAGE_NAME)

if(NOT INDICATOR_HEADER_WITH_PATH_PREFIX)
  message(FATAL_ERROR "ERROR: In SimplePackageFind, ${PACKAGE_NAME} INDICATOR_HEADER_WITH_PATH_PREFIX input parameter is empty!")
endif(NOT INDICATOR_HEADER_WITH_PATH_PREFIX)

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
if(NOT LIBRARY_SEARCH_NAME STREQUAL "")
    find_library_multipass(${PACKAGE_NAME}_LIB ${LIBRARY_SEARCH_NAME})

    # Report status
    if("${${PACKAGE_NAME}_LIB}" MATCHES "-NOTFOUND$")
      message(STATUS "Looking for lib${LIBRARY_SEARCH_NAME} - MISSING")
    else()
      message(STATUS "Looking for lib${LIBRARY_SEARCH_NAME} - found")
    endif()
else()
    set(${PACKAGE_NAME}_LIB "UNNEEDED")
endif()

# Ensure that the value of this variable is transmitted to a higher scope
set(${PACKAGE_NAME}_LIB ${${PACKAGE_NAME}_LIB} PARENT_SCOPE)

######## Final processing ########

# These two lines use a standard part of CMake to provide the basic
# variable defines and messages expected from a Find___ module.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(${PACKAGE_NAME}
    REQUIRED_VARS "${PACKAGE_NAME}_INCLUDE_DIR" "${PACKAGE_NAME}_LIB")

# Mark these as "advanced" variables, so they only show up when doing
# "advanced" CMake configuration
mark_as_advanced("${PACKAGE_NAME}_INCLUDE_DIR" "${PACKAGE_NAME}_LIB")

# Generate imported targets
if(${PACKAGE_NAME}_FOUND AND NOT TARGET ${PACKAGE_NAME}::${PACKAGE_NAME})
  if(NOT LIBRARY_SEARCH_NAME STREQUAL "")
    # Create an imported target of the correct type
    if(${PACKAGE_NAME}_LIB MATCHES "[.](so|dylib)$")
      add_library(${PACKAGE_NAME}::${PACKAGE_NAME} SHARED IMPORTED)
    elseif(${PACKAGE_NAME}_LIB MATCHES "[.](a|o)$")
      add_library(${PACKAGE_NAME}::${PACKAGE_NAME} STATIC IMPORTED)
    else() # e.g. .dll is used for both shared and static libs on Windows
      add_library(${PACKAGE_NAME}::${PACKAGE_NAME} UNKNOWN IMPORTED)
    endif()
    set_target_properties(${PACKAGE_NAME}::${PACKAGE_NAME} PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX")
  else()
    # Header-only packages
    add_library(${PACKAGE_NAME}::${PACKAGE_NAME} INTERFACE IMPORTED)
  endif()
  set_target_properties(${PACKAGE_NAME}::${PACKAGE_NAME} PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${${PACKAGE_NAME}_INCLUDE_DIR}"
    )
  if((NOT LIBRARY_SEARCH_NAME STREQUAL "")
    AND (NOT ${${PACKAGE_NAME}_LIB} STREQUAL "")
    AND (NOT ${${PACKAGE_NAME}_LIB} MATCHES "-NOTFOUND$"))
    set_target_properties(${PACKAGE_NAME}::${PACKAGE_NAME} PROPERTIES
      IMPORTED_LOCATION "${${PACKAGE_NAME}_LIB}"
      )
  endif()
endif()

endfunction(SimplePackageFind)
