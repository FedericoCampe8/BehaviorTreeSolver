#########################################
# Define system tests
#########################################

# This is a list of all system tests
set(sys_tests systest.test)

#########################################
# Tell Cmake how to run the system tests
#########################################

# This iterates over all system tests and defines tests for them
# (this is assuming they're all just scripts; this can get more complicated if
# the tests need to be compiled - it will look more like the unit test
# CMakeLists.txt file)
foreach(sys_test IN LISTS sys_tests)
    add_test(NAME sys_${sys_test}
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/${sys_test}" one two three)
endforeach()
