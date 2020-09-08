# This file only includes truly "basic" setup, such as
# setting up generic targets (e.g. "release"), and similar simple tasks.

include(ColorMessage)

###### SET RPATHS ON MAC ARCHITECTURES ######
# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

set(LIB_DLINK_PATH "")
if( UNIX AND NOT APPLE )
  set(LIB_DLINK_PATH "linux")
elseif( APPLE )
  set(LIB_DLINK_PATH "mac")
endif()

set(LIBS_DLINK_FULL_PATH "${PROJECT_SOURCE_DIR}/lib/${lib_arch_dir}")
set(CMAKE_INSTALL_RPATH "${LIBS_DLINK_FULL_PATH}/protobuf_3_7_1")
list(APPEND CMAKE_INSTALL_RPATH "${LIBS_DLINK_FULL_PATH}/or-tools")
list(APPEND CMAKE_INSTALL_RPATH "${LIBS_DLINK_FULL_PATH}/ssl")
list(APPEND CMAKE_INSTALL_RPATH "${LIBS_DLINK_FULL_PATH}/zeromq")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")
############################################

# If you want to override one of the options (such as PERFORM_TESTS),
# you need to set the value in the cache. If you don't, CMake
# will largely ignore your setting as soon as the cached option is defined. The
# right way to override it in your own CMakeLists.txt file is:
#     set(PERFORM_TESTS OFF CACHE BOOL "Disabled by fiat")

# This must be defined early, because its used several places
option(PERFORM_TESTS "Flag to enable/disable building unit tests" ON)
# option(CODE_COVERAGE "Unset if debug builds should not support code coverage" ON)
option(SANITIZE_ADDRESS "Flag to enable/disable the use of the compiler's address sanitizer. Defaults to ON in Debug, and is forced OFF in Release mode." ON)
option(CPPCHECK "Set if static analysis of code should be performed." Off)


if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  # Address sanitizing is NOT allowed in release builds
  set(SANITIZE_ADDRESS OFF CACHE BOOL "Flag to enable/disable the use of the compiler's address sanitizer. Defaults to ON in Debug, and is forced OFF in Release mode." FORCE)
endif()

if(NOT PERFORM_TESTS)
    set(CODE_COVERAGE OFF CACHE BOOL "Disabled because tests are disabled" FORCE)
endif()

# This is a standard CMake option, Added here to provide control over the
# default behavior
option(BUILD_SHARED_LIBS "Flag to toggle between shared/static libraries" ON)

if(NOT "${ENABLE_CODE_COVERAGE}" STREQUAL "")
  set(CODE_COVERAGE ${ENABLE_CODE_COVERAGE} CACHE BOOL "Inherited from ENABLE_CODE_COVERAGE" FORCE)
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  # This is the brazil-friendly default
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Location of installation" FORCE)
endif()

# These are STRING type, not PATH, because they may not exist yet (and are
# relative to CMAKE_INSTALL_PREFIX if they're not absolute paths)
set(INSTALL_INCLUDEDIR "include" CACHE STRING "Location where headers will be installed")
set(INSTALL_LIBDIR "lib" CACHE STRING "Location where libraries will be installed")
set(INSTALL_BINDIR "bin" CACHE STRING "Location where executables will be installed")
message(STATUS "Installing to ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  -- Headers: ${INSTALL_INCLUDEDIR}")
message(STATUS "  -- Libs   : ${INSTALL_LIBDIR}")
message(STATUS "  -- Exe    : ${INSTALL_BINDIR}")

####################################################################
# Define useful macros
####################################################################
include(AddCXXFlagMacros)

####################################################################
# Retrieve environment compiler flags, instead of relying on the cache
####################################################################
if(RELOAD_ENVIRONMENT)
    set(CMAKE_C_FLAGS $ENV{CFLAGS} CACHE STRING "Compiler flags for C code" FORCE)
    set(CMAKE_CXX_FLAGS $ENV{CXXFLAGS} CACHE STRING "Compiler flags for C++ code" FORCE)
    set(CMAKE_EXE_LINKER_FLAGS_INIT $ENV{LDFLAGS} CACHE STRING "Linker flags for executables" FORCE)
    set(CMAKE_SHARED_LINKER_FLAGS_INIT $ENV{LDFLAGS} CACHE STRING "Linker flags for shared libraries" FORCE)
    set(CMAKE_MODULE_LINKER_FLAGS_INIT $ENV{LDFLAGS} CACHE STRING "Linker flags for modules (dlopen-able)" FORCE)
    message(STATUS "C++ Flags Pulled from Environment: ${CMAKE_CXX_FLAGS}")
endif()

####################################################################
# Setup standard compiler flags
####################################################################
# Note that the Add___Flag macros won't add the flag if the compiler or linker
# complain

AddCXXFlag("-fcolor-diagnostics;-fdiagnostics-color=always")
AddCXXFlag("-Werror")

# The -pipe flag, supported on all systems that use the GNU assembler, tells
# the compiler to avoid temporary files and to use pipes instead. It should
# speed up builds
AddCXXFlag("-pipe")

# The --as-needed flag is a backstop against unnecessary libraries. It's
# generally harmless, but avoids linking against libraries that are not
# actually referenced.
# For more information on the utility of the as-needed flag, see here:
# https://wiki.gentoo.org/wiki/Project:Quality_Assurance/As-needed
AddLinkerFlag("--as-needed")

# The --no-undefined flag checks that the shared library you're building will
# not end up with symbols that are not resolved by any of its dependencies.
# This means you can discover missing symbols earlier (i.e. when the shared
# libary is created, rather than when it is used), which leads to better
# diagnostic information.
AddLinkerFlag("--no-undefined")

# This is similar to --no-undefined, but performs the same check on all
# libraries being linked in. Thus, we can catch problems where a dependency is
# built improperly before they get propagated to our customers.
AddLinkerFlag("--no-allow-shlib-undefined")

# The linker can (with this flag) optimize the symbol table size; doing so
# reduces the number of hash collisions, which speeds up symbol resolution in
# shared libraries.
AddLinkerFlag("-O1")

# This tells the linker to use the newer GNU-style symbol table, which
# dramatically reduces the number of string comparisons necessary during symbol
# resolution when using shared libraries.
AddLinkerFlag("--hash-style=gnu")

if(SANITIZE_ADDRESS)
  # These two flags are tied together, and have the effect of doing a
  # compiler-assisted valgrind run. The performance impact is roughly 2x
  # slower, and anything that links to this needs to be built in the same way
  # (otherwise you get weird segfaults, because this plays games with memory
  # layout). This defaults to ON when doing a debug build.
  AddLinkerFlag("-fsanitize=address")
  AddCXXFlag("-fsanitize=address")
endif()

####################################################################
# Include standard modules
####################################################################
# This standard CMake package is very helpful for generating export macros
# portably (rather than copying the header into every package). See
# https://cmake.org/cmake/help/v3.12/module/GenerateExportHeader.html for more
# information.
include(GenerateExportHeader)

# Set up compiler flags for code coverage
# include(CodeCoverageSettings)

# A macro for making Boost easier to work with in a Brazil environment
include(BoostFinder)

# CppCheck compatibility
# include(CppCheckSetup)

####################################################################
# Establish key targets
####################################################################

# Key targets: release, debug
#
# These targets will reconfigure the build to either Release or Debug
# BUILD_TYPEs (thus changing the compiler flags in use) if necessary. These are
# convenience targets
#
# The release target ensures that:
#  - Unit/System tests get run (via dependency on 'check' target)
#  - The CMAKE_CURRENT_BINARY_DIR is built with whatever the correct build system is
#  - The installable files get installed (the "install" target is triggered)
add_custom_target(release
    COMMENT "Default target when building a VersionSet (or doing a Coverlay build)"
    USES_TERMINAL
    COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}" --target coverage
    COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}" --target clean
    COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}" --target optimized_install
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

add_custom_target(optimized_install
    COMMENT "Build and install optimized binaries"
    USES_TERMINAL
    COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}" --target install
    DEPENDS check
    )

add_custom_target(debug
    COMMENT "Building debugging-enabled binaries"
    USES_TERMINAL
    COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}"
    )

add_custom_target(deep-clean
    COMMENT "Deleting all temporary build products"
    COMMAND echo rm -rf "${CMAKE_BINARY_DIR}/")

if(NOT PERFORM_TESTS)
    message(STATUS "All tests are DISABLED")
    add_custom_target(check
        COMMENT "Skipping unit and system tests")
endif()

if(NOT CODE_COVERAGE)
    message(STATUS "Code coverage collection DISABLED")
    add_custom_target(coverage
        COMMENT "Skipping code coverage collection")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    cmessage(GREEN "==================================================")
    cmessage(GREEN "===================={ RELEASE }===================")
    cmessage(GREEN "==================================================")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    cmessage(YELLOW "==================================================")
    cmessage(YELLOW "===================={ DEBUG }=====================")
    cmessage(YELLOW "==================================================")
else()
    cmessage(YELLOW "==================================================")
    cmessage(YELLOW "==================={ ${CMAKE_BUILD_TYPE} }====================")
    cmessage(YELLOW "==================================================")
endif()

# include(CppLintTarget)

# vim: sw=4 expandtab
