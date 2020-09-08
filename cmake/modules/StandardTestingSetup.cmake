include(BasicTestSetup)

# Using full-path because it's required to get well-defined behavior
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/unit")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/unit")
endif()
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/system")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/system")
endif()
