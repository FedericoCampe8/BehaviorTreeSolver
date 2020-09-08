set(CTEST_FLAGS "" CACHE STRING "Arguments that are passed to CTest when running the test suite.")
set(CTEST_UNIT_FLAGS "" CACHE STRING "Arguments that are passed to CTest when running the unit test suite.")
set(CTEST_SYS_FLAGS "" CACHE STRING "Arguments that are passed to CTest when running the unit test suite.")

option(PARALLEL_TESTS "Allows tests to run in parallel, if they're not required to be serial" Off)
if(NOT "${ENABLE_PARALLEL_TESTS}" STREQUAL "")
  set(PARALLEL_TESTS ${ENABLE_PARALLEL_TESTS} CACHE BOOL "Inherited from ENABLE_PARALLEL_TESTS" FORCE)
endif()

if(PARALLEL_TESTS)
    # This allows ctest tests (which are run by ctest) to run in parallel
    include(ProcessorCount)
    ProcessorCount(N) # Conveniently, N can only ever be positive
    if(NOT N EQUAL 0)
        message(STATUS "Detected ${N} processors.")
        list(APPEND CTEST_FLAGS "-j${N}")
    endif()
endif(PARALLEL_TESTS)

list(APPEND CTEST_FLAGS "--output-on-failure")

add_custom_target(build_everything
    COMMENT "Building everything, so that tests can be run"
    USES_TERMINAL
    COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}"
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

add_custom_target(check
    COMMENT "Running unit and system tests"
    USES_TERMINAL
    DEPENDS build_everything
    COMMAND ${CMAKE_CTEST_COMMAND} ${CTEST_FLAGS}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

add_custom_target(checkunit
    COMMENT "Running unit tests"
    USES_TERMINAL
    DEPENDS build_everything
    COMMAND ${CMAKE_CTEST_COMMAND} ${CTEST_FLAGS} ${CTEST_UNIT_FLAGS} -R 'unit_*'
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

add_custom_target(checksys
    COMMENT "Running system tests"
    USES_TERMINAL
    DEPENDS build_everything
    COMMAND ${CMAKE_CTEST_COMMAND} ${CTEST_FLAGS} ${CTEST_SYS_FLAGS} -R 'sys_*'
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

# vim: sw=4 expandtab
