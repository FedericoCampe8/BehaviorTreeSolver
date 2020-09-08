include(PackageMacros)

# Generating the correct tree of cpp files
set(source_blackmasks "")
do_file_list(bt_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpp" ".cpp" "${source_blackmasks}" "${exe_file_list}")
