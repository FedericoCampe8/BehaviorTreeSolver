#########################################
# Define unit tests
#########################################

include(PackageMacros)

# This is the set of sources for the basic test
set(unit_test_SOURCES "")
# list(APPEND unit_test_SOURCES "${PROTO_SRCS}")

set(source_blackmasks "")
set(source_blacklist "")
do_file_list(unit_test_SOURCES "${PROJECT_SOURCE_DIR}/test/unit" ".cpp" "${source_blackmasks}" "${source_blacklist}")

# This is the set of sources for the basic test
#set(unit_test_SOURCES
#    config/config_test.cpp
#    )

#########################################
# Tell Cmake how to run the unit tests
#########################################

# This defines each unit test and associates it with its sources
add_executable(unitTests ${unit_test_SOURCES})

# This links the unit_test executable against the main library
target_link_libraries(unitTests BTSolver gtest_main)
#target_link_libraries(unitTests BTSolver gmock_main)

# This adds the right gtest dependencies (macro defined by the Gtest setup
# modules in cmake folder)
add_gtest_to_target(unitTests)
#add_gmock_to_target(unitTests)

# This tells CTest about this unit test executable
# The TEST_PREFIX prepends "unit_" to the name of these tests in the output.
gtest_discover_tests(unitTests
                     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                     TEST_PREFIX unit_
                     TEST_LIST unit_gtests)

# This tells the CTest harness about how it should treat these tests
set_tests_properties(${unit_gtests} PROPERTIES
    RUN_SERIAL 1
    TIMEOUT 30 # seconds
    LABELS UnitTests)
