# Specify the minimum CMAKE version
cmake_minimum_required(VERSION 3.12.0 FATAL_ERROR)

# Define the project
project(BehaviorTreeSolver VERSION 1.0 LANGUAGES C CXX)

# Set c++ version and make it a requirement when building the targets
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the include directory in src to the search path for include files.
# Note: PROJECT_SOURCE_DIR is the source directory of the project()
# command above. Same for PROJECT_BINARY_DIR w.r.t. the binary directory.
# This defines the primary include list
include_directories(
  "${PROJECT_SOURCE_DIR}/cpp"
  "${PROJECT_SOURCE_DIR}/third-party"
  "${PROJECT_SOURCE_DIR}/third-party/sparsepp"
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_BINARY_DIR}/src"
)

# The following sets a variables for excluding files from cpplint checks
# list(APPEND CPPLINT_EXCLUDED_FILES src/some_file.cc)

# Note: to print information during cmake builds use "message".
# For example:
# message( "Variables: " ${bt_SOURCES} )

#########################################################################
# BASIC SETUP:
# - set the variable for the modules (cmake) folders
# - run the basic setup module
#########################################################################
# Set variable for the makefiles folder.
# It is possible to define this variable by running cmake with the same
# variable as argument as follows:
# -DMAKE_FILES_COMMON=/path/to/makefiles
set(DMAKE_FILES_COMMON "${PROJECT_SOURCE_DIR}/cmake" CACHE STRING "Setting to specify the location of the cmake folder")

# Check for the variable to be set
if("${DMAKE_FILES_COMMON}" STREQUAL "")
  message(FATAL_ERROR "Unspecified location for makefiles folder. Please define this variable manually (e.g., run cmake with -DMAKE_FILES_COMMON=/path/to/cmake).")
endif()

# Search macros, variables, and other makefiles when looking for modules
list(APPEND CMAKE_MODULE_PATH ${DMAKE_FILES_COMMON}/modules)

# Load and run CMake code from the BasicSetup module
include(BasicSetup)
#########################################################################

# Show all warnings when compiling
#AddCXXFlag("-Wall")
AddCXXFlag("-Wunused-function")
AddCXXFlag("-Wno-deprecated")
AddCXXFlag("-Wno-deprecated-declarations")

# Hide symbols by default (e.g. on GCC to use -fvisibility=hidden).
# This should be overridden in API headers.
# Practically, choose which functions ones want to be visible to users linking
# against the library and make them visible by marking them with a
# visible attribute.
# For example:
# void __attribute__((visibility("default"))) Exported()
# {
#   // ...
# }
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# Hide inline symbols (e.g. on GCC to use -fvisibility-inlines-hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

############################
# Add library dependencies #
############################

# Set global path to third party libraries depending on
# the current architecture
set(lib_arch_dir "")
if( UNIX AND NOT APPLE )
  set(lib_arch_dir "linux")
elseif( APPLE )
  set(lib_arch_dir "mac")
endif()
set(THIRD_PARTY_LIBS_PATH "${PROJECT_SOURCE_DIR}/lib/${lib_arch_dir}")

# Boost
# To set a specific location for boost, do the following:
# set(BOOST_LIBRARYDIR /usr/local/opt/boost/lib64)
#find_boost(system chrono filesystem regex thread timer)
#link_libraries(
#	"${Boost_TIMER_LIBRARY}"
# "${Boost_THREAD_LIBRARY}"
#	"${Boost_FILESYSTEM_LIBRARY}"
#	"${Boost_SYSTEM_LIBRARY}"
# "${Boost_CHRONO_LIBRARY}"
# "${Boost_REGEX_LIBRARY}"
#	)

#find_package(Boost REQUIRED COMPONENTS boost_system boost_chrono boost_filesystem boost_regex boost_thread boost_timer)
# Treat Boost as system libraries, to suppress warnings
#include_directories(BEFORE SYSTEM "${Boost_INCLUDE_DIRS}")
#include_directories(BEFORE SYSTEM ${Boost_INCLUDE_DIR})
#link_libraries(${Boost_LIBS})

# DL libs
link_libraries(${CMAKE_DL_LIBS})

# pthreads and standard math library links on Linux architectures
if (UNIX AND NOT APPLE)
  # Threads
  set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
  set(THREADS_PREFER_PTHREAD_FLAG TRUE)
  find_package(Threads REQUIRED)
  link_libraries(${CMAKE_THREAD_LIBS_INIT})
  link_libraries(pthread)

  # Standard math libraries
  find_package(StandardMathLibrary)
  if(NOT STANDARD_MATH_LIBRARY_FOUND)
    message(FATAL_ERROR "Can't link to the standard math library.")
  else()
    link_libraries(${STANDARD_MATH_LIBRARY})
    link_libraries(m)
    link_libraries(z)
  endif()
endif()

#######################
# Define main library #
#######################

# Create a variable to keep track of the source files and one
# to keep track of the executable files
set(bt_SOURCES "")
set(exe_file_list "bt_client/bt_client.cpp" "mdd_client/mdd_client.cpp")

# Create a variable to track the sources of the main target.
# Note: the following include will call a macro that recursively looks
# for all the cpp files.
# This can be manually accomplished using the following command:
# set(bt_SOURCES cpp/path_to_file/file.cpp)
include(MainSrcFinder)

# Remember to add protobuf files
list(APPEND bt_SOURCES "${PROTO_SRCS}")
# list(APPEND bt_SOURCES "${PROTO_HDRS}")

# Define the primary library and version
add_library(BTSolver SHARED ${bt_SOURCES})
set_target_properties(BTSolver PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(BTSolver PROPERTIES SOVERSION 1)

# Declare libraries files to install
# INSTALL_INCLUDEDIR is defined by the BasicSetup module
# This statement installs the library built by the specified target
install(TARGETS BTSolver
        LIBRARY DESTINATION ${INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${INSTALL_LIBDIR}
        FRAMEWORK DESTINATION ${INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDEDIR}")

# Install all headers in the specified directory.
# If there are multiple headers, it is recommended to have a
# subdirectory with the name of the directory, e.g. cpp/utilities/, so that
# consuming packages can use the headers as follows:
#   #include <utilities/header.hpp>
# Such a directory would not be reflected here.
# This statement is along the lines of:
#   cp -a cpp/utilities/* ${INSTALL_INCLUDEDIR}
install(DIRECTORY cpp
        DESTINATION "${INSTALL_INCLUDEDIR}"
        FILES_MATCHING PATTERN "*.hpp")

# This must be here to properly enable the CMake testing infrastructure
if(PERFORM_TESTS)
  # enable_testing() MUST be called before adding the test subdirectory in
  # order to get reliable CTest behavior (failing to do so may make CTest
  # think there are no tests). Also, this directory should be added AFTER
  # dependent libraries are discovered (e.g. Boost)
  enable_testing()
  add_subdirectory(test)
endif()

# Set target directory for the executable
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
#add_executable(bt_solver cpp/bt_client/bt_client.cpp)
#target_link_libraries(bt_solver LINK_PUBLIC BTSolver)

add_executable(mdd_solver cpp/mdd_client/mdd_client.cpp)
target_link_libraries(mdd_solver LINK_PUBLIC BTSolver)

# Install the target executable into the specified folder
# install(TARGETS bt_solver RUNTIME DESTINATION "${INSTALL_BINDIR}")
install(TARGETS mdd_solver RUNTIME DESTINATION "${INSTALL_BINDIR}")
