# Creates list of files, scanning subdirectories starting from path.
# Files string must end with "endmask".
# All files containing "blackmask" will be ignored.
# All files contained in "blacklist" will be ignored.
macro(do_file_list ext_list path endmask blackmask blacklist)
    set(inFiles "")
    FILE(GLOB_RECURSE inFiles RELATIVE "${path}"
        "${path}/*${endmask}")

    set(blck "${blacklist}")
    FOREACH(infileName ${inFiles})
        if (NOT ${infileName} IN_LIST blck)
            if ("${blackmask}" STREQUAL "")
                set(${ext_list} ${${ext_list}} "${path}/${infileName}")
            else()
                #excluding every possible blackmask
                SET( valid 1 )
                FOREACH(blackmask_iteration ${blackmask})
                    if(${infileName} MATCHES "${blackmask_iteration}")
                        SET( valid 0 )
                    endif()
                ENDFOREACH(blackmask_iteration)
                if(valid EQUAL 1)
                    set(${ext_list} ${${ext_list}} "${path}/${infileName}")
                endif() #if 0 string is not included in results
            endif()
        endif()
    ENDFOREACH(infileName)
endmacro()

macro(check_directory VAR_TO_CHECK TYPE_OF_DIR PACKAGE_NAME SUCCESSVAR)
  if(${VAR_TO_CHECK})
    if(NOT EXISTS ${${VAR_TO_CHECK}})
      cmessage(YELLOW " - Specified ${TYPE_OF_DIR} for ${PACKAGE_NAME} does not exist! ${${VAR_TO_CHECK}}")
    else()
      set(${SUCCESSVAR} ${${VAR_TO_CHECK}})
    endif()
  endif()
endmacro()

macro(process_standard_input_variables PACKAGE_NAME)
  string(TOUPPER ${PACKAGE_NAME} UPPER_PACKAGE_NAME)
  check_directory(${UPPER_PACKAGE_NAME}_ROOT
                  "root"
                  ${PACKAGE_NAME}
                  ${PACKAGE_NAME}_ROOT_)
  check_directory(${PACKAGE_NAME}_ROOT
                  "root"
                  ${PACKAGE_NAME}
                  ${PACKAGE_NAME}_ROOT_)
  if(${PACKAGE_NAME}_ROOT_)
    message(STATUS "Using defined root for ${PACKAGE_NAME}: ${${PACKAGE_NAME}_ROOT_}")
    set(${PACKAGE_NAME}_INCLUDE_PATH "${${PACKAGE_NAME}_ROOT_}/include")
    set(${PACKAGE_NAME}_LIB_PATH "${${PACKAGE_NAME}_ROOT_}/lib")
  endif()

  check_directory(${UPPER_PACKAGE_NAME}_INCLUDE_PATH
                  "header search directory"
                  ${PACKAGE_NAME}
                  ${PACKAGE_NAME}_INCLUDE_PATH_)
  check_directory(${PACKAGE_NAME}_INCLUDE_PATH
                  "header search directory"
                  ${PACKAGE_NAME}
                  ${PACKAGE_NAME}_INCLUDE_PATH_)
  if(${PACKAGE_NAME}_INCLUDE_PATH_)
    set(INCLUDE_PATH "${${PACKAGE_NAME}_INCLUDE_PATH_}")
    if(NOT ${PACKAGE_NAME}_ROOT_)
      message(STATUS "Using defined include path for ${PACKAGE_NAME}: ${INCLUDE_PATH}")
    endif()
  endif()

  check_directory(${UPPER_PACKAGE_NAME}_LIB_PATH
                  "library search directory"
                  ${PACKAGE_NAME}
                  ${PACKAGE_NAME}_LIB_PATH_)
  check_directory(${PACKAGE_NAME}_LIB_PATH
                  "library search directory"
                  ${PACKAGE_NAME}
                  ${PACKAGE_NAME}_LIB_PATH_)
  if(${PACKAGE_NAME}_LIB_PATH_)
    set(LIB_PATH "${${PACKAGE_NAME}_LIB_PATH_}")
    if(NOT ${PACKAGE_NAME}_ROOT_)
      message(STATUS "Using defined library path for ${PACKAGE_NAME}: ${LIB_PATH}")
    endif()
  endif()

  if(INCLUDE_PATH)
    set(INCLUDE_DEFAULT_PATH_OVERRIDE "NO_DEFAULT_PATH")
  endif()

  if(LIB_PATH)
    set(LIB_DEFAULT_PATH_OVERRIDE "NO_DEFAULT_PATH")
  endif()
endmacro()

macro(find_header_multipass)
  # We want to prioritize manually-specified paths over
  # automatically-determined paths such as system paths, so we do this in two
  # passes. The first pass searches ONLY manually specified paths.
  # The second pass searches ONLY automatically-determined paths.

  # First pass:
  find_path(${PACKAGE_NAME}_INCLUDE_DIR_FIRSTPASS ${INDICATOR_HEADER_WITH_PATH_PREFIX}
    PATHS "${INCLUDE_PATH}"
    "${INCLUDE_PATH}/../${INSTALL_INCLUDEDIR}"
    PATH_SUFFIXES ${PATH_SUFFIX}
    NO_DEFAULT_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH)
  mark_as_advanced(${PACKAGE_NAME}_INCLUDE_DIR_FIRSTPASS)

  if("${${PACKAGE_NAME}_INCLUDE_DIR_FIRSTPASS}" MATCHES "-NOTFOUND$")
    # If the first scan failed, try the system/default paths
    # Second pass
    find_path(${PACKAGE_NAME}_INCLUDE_DIR_SECONDPASS ${INDICATOR_HEADER_WITH_PATH_PREFIX}
      PATH_SUFFIXES ${PATH_SUFFIX}
      "${INCLUDE_DEFAULT_PATH_OVERRIDE}")
    mark_as_advanced(${PACKAGE_NAME}_INCLUDE_DIR_SECONDPASS)
    # Copy second-pass result (whatever it is) to the desired *_INCLUDE_DIR variable
    set(${PACKAGE_NAME}_INCLUDE_DIR ${${PACKAGE_NAME}_INCLUDE_DIR_SECONDPASS}
      CACHE PATH "Directory containing ${INDICATOR_HEADER_WITH_PATH_PREFIX}")
  else()
    # Copy first-pass success to the desired *_INCLUDE_DIR variable
    set(${PACKAGE_NAME}_INCLUDE_DIR ${${PACKAGE_NAME}_INCLUDE_DIR_FIRSTPASS}
      CACHE PATH "Directory containing ${INDICATOR_HEADER_WITH_PATH_PREFIX}")
  endif()
  # By this point, ${${PACKAGE_NAME}_INCLUDE_DIR} either has the directory, or
  # something that ends in -NOTFOUND

  # Indicate via a message to the operator whether the indicator file was found or not
  if("${${PACKAGE_NAME}_INCLUDE_DIR}" MATCHES "-NOTFOUND$")
    message(STATUS "Looking for ${INDICATOR_HEADER_WITH_PATH_PREFIX} - MISSING")
  else()
    # Unfortunately, CMake doesn't treat the path-prefix as mandatory, and
    # there's no way (as of CMake 3.12) to make it mandatory. Therefore, we
    # have to check (and potentially fail) manually.
    set(target_path
      ${${PACKAGE_NAME}_INCLUDE_DIR}/${INDICATOR_HEADER_WITH_PATH_PREFIX})
    if(NOT EXISTS "${target_path}")
      # Now remove the path suffix portion of the full path to the indicator
      # file so that the include filepath for this package points at the
      # top-most level desired (i.e. #includes can look like
      # "#include <a/b/file.h>")
      while(INDICATOR_FILE_DIR)
        # Strip off directory levels one by one until the INDICATOR_FILE_DIR is
        # gone from the full path
        get_filename_component(${PACKAGE_NAME}_INCLUDE_DIR
          ${${PACKAGE_NAME}_INCLUDE_DIR} DIRECTORY)
        get_filename_component(INDICATOR_FILE_DIR
          ${INDICATOR_FILE_DIR} DIRECTORY)
      endwhile(INDICATOR_FILE_DIR)
      set(target_path
        ${${PACKAGE_NAME}_INCLUDE_DIR}/${INDICATOR_HEADER_WITH_PATH_PREFIX})
      if(NOT EXISTS "${target_path}")
        set(${PACKAGE_NAME}_INCLUDE_DIR "${PACKAGE_NAME}_INCLUDE_DIR-NOTFOUND")
      endif()
    endif()

    if("${${PACKAGE_NAME}_INCLUDE_DIR}" MATCHES "-NOTFOUND$")
      message(STATUS "Looking for ${INDICATOR_HEADER_WITH_PATH_PREFIX} - NOT FOUND")
    else()
      message(STATUS "Looking for ${INDICATOR_HEADER_WITH_PATH_PREFIX} - found")
    endif()
  endif()
endmacro(find_header_multipass)

macro(find_library_multipass VAR_BASE LIBNAMES)
  # We want to prioritize manually-specified paths over
  # automatically-determined paths such as system paths, so we do this in two
  # passes. The first pass searches ONLY manually specified paths.
  # The second pass searches ONLY automatically-determined paths.

  # components may have multiple names for the library, which must be passed in
  # as a CMake list (i.e. semicolon-delimited)

  # First pass:
  foreach(LIBNAME ${LIBNAMES})
    find_library(${VAR_BASE}_${LIBNAME}_FIRSTPASS ${LIBNAME}
      PATHS "${LIB_PATH}"
      "${LIB_PATH}/../${INSTALL_LIBDIR}"
      PATH_SUFFIXES ${PATH_SUFFIX}
      NO_DEFAULT_PATH
      NO_CMAKE_ENVIRONMENT_PATH
      NO_CMAKE_PATH
      NO_SYSTEM_ENVIRONMENT_PATH
      NO_CMAKE_SYSTEM_PATH)
    mark_as_advanced(${VAR_BASE}_${LIBNAME}_FIRSTPASS)
    if(NOT "${${VAR_BASE}_${LIBNAME}_FIRSTPASS}" MATCHES "-NOTFOUND$")
      set(${VAR_BASE}_FIRSTPASS ${${VAR_BASE}_${LIBNAME}_FIRSTPASS} CACHE PATH "")
      break()
    endif()
  endforeach()

  if("${${VAR_BASE}_FIRSTPASS}" MATCHES "-NOTFOUND$")
    # If the first scan failed, try the system/default paths
    # Second pass
    foreach(LIBNAME ${LIBNAMES})
      find_library(${VAR_BASE}_${LIBNAME}_SECONDPASS ${LIBNAME}
        PATH_SUFFIXES ${PATH_SUFFIX}
        "${INCLUDE_DEFAULT_PATH_OVERRIDE}")
      mark_as_advanced(${VAR_BASE}_${LIBNAME}_SECONDPASS)
      if(NOT "${${VAR_BASE}_${LIBNAME}_SECONDPASS}" MATCHES "-NOTFOUND$")
        # Copy second-pass result (whatever it is) to the desired variable
        set(${VAR_BASE} ${${VAR_BASE}_${LIBNAME}_SECONDPASS} CACHE PATH "")
        break()
      endif()
    endforeach()
  else()
    # Copy first-pass success to the desired variable
    set(${VAR_BASE} ${${VAR_BASE}_FIRSTPASS} CACHE PATH "")
  endif()
  # By this point, ${VAR_BASE} either has the directory, or something that ends
  # in -NOTFOUND
endmacro()
