# This file includes a few convenience functions, for safely adding compiler flags
# only if the compiler supports them

include(CheckCXXCompilerFlag)

# This function converts the specified flag into a C identifier (converts hyphens
# and other inconvenient characters to underscores), stored in a variable named
# "safe_flag_name" in the local scope, and then checks whether the compiler
# supports that flag (using the CMake-standard CHECK_CXX_COMPILER_FLAG
# function). This test is cached in the COMPILER_SUPPORTS_${safe_flag_name}
# variable.
function(DoesCompilerSupportFlag lang flag_to_test output_var)
    string(MAKE_C_IDENTIFIER "${flag_to_test}" c_identifier)
    if (${lang} STREQUAL "CXX")
        CHECK_CXX_COMPILER_FLAG(${flag_to_test} "${lang}_COMPILER_SUPPORTS_${c_identifier}")
    elseif(${lang} STREQUAL "C")
        CHECK_C_COMPILER_FLAG(${flag_to_test} "${lang}_COMPILER_SUPPORTS_${c_identifier}")
    else()
        MESSAGE(FATAL_ERROR "Language parameter in DoesCompilerSupportFlag must be CXX or C, but it is ${lang}")
    endif()
    set(${output_var} ${c_identifier} PARENT_SCOPE)
    set(${lang}_COMPILER_SUPPORTS_${c_identifier} ${${lang}_COMPILER_SUPPORTS_${c_identifier}} PARENT_SCOPE)
endfunction()

# This function is very similar to the DoesCompilerSupportCXXFlag function, but
# exists to work around the fact that CHECK_CXX_COMPILER_FLAG is intended just
# for compiler flags. The fact is that CMake uses the compiler as the linker,
# rather than invoking the linker directly. This test is cached in the
# COMPILER_SUPPORTS_${safe_flag_name} variable.
function(DoesLinkerSupportFlag flag_to_test c_identifier)
    # Convert to linker flag
    # exe linker flags are what are used by the CHECK_CXX_COMPILER_FLAG function
    set(SAVED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${flag_to_test}")
    CHECK_CXX_COMPILER_FLAG("" "LINKER_SUPPORTS_${c_identifier}")
    set(CMAKE_EXE_LINKER_FLAGS "${SAVED_LINKER_FLAGS}")
    # report results
    set(LINKER_SUPPORTS_${c_identifier} ${LINKER_SUPPORTS_${c_identifier}} PARENT_SCOPE)
endfunction()

macro(AddFlagInternal lang flaglist success_var)
    set(${success_var} OFF)
    foreach(flag_under_test ${flaglist})
        if("${flag_under_test}" MATCHES "-D.*")
            add_definitions(${flag_under_test})
            set(${success_var} ON)
            break()
        endif()
        DoesCompilerSupportFlag("${lang}" ${flag_under_test} safe_flag_name)
        if(${${lang}_COMPILER_SUPPORTS_${safe_flag_name}})
            message(STATUS "Adding ${lang} Compiler flag: ${flag_under_test} to ALL targets declared after this point, in subdirectories of ${CURRENT_SOURCE_DIR}")
            add_compile_options($<$<COMPILE_LANGUAGE:${lang}>:${flag_under_test}>)
            set(${success_var} ON)
            break()
        endif()
    endforeach()
endmacro()

### ------------------------------ ###
### -------- CXX FLAGS ----------- ###
### ------------------------------ ###
# This macro checks the flags in the flaglist, and adds the first one that
# is accepted by the compiler to the compile options for all subsequent targets
# (via add_compile_options)
function(AddOptionalCXXFlag flaglist)
    AddFlagInternal("CXX" "${flaglist}" FAILURE_IS_IGNORED)
endfunction()

# Synonym of AddOptionalCXXFlag
function(AddCXXFlag flaglist)
    AddFlagInternal("CXX" "${flaglist}" FAILURE_IS_IGNORED)
endfunction()

# This macro checks the flags in the flaglist, and adds the first one that
# is accepted by the compiler to the compile options for all subsequent targets
# (via add_compile_options). If none are supported, it fails the configuration.
function(AddRequiredCXXFlag flaglist)
    AddFlagInternal("CXX" "${flaglist}" SUCCESS_CHECK)
    if(NOT SUCCESS_CHECK)
        string(REPLACE ";" "\n\t" optionlist "${flaglist}")
        cmessage(FATAL_ERROR "None of the desired flag alternatives are supported by the compiler:\n\t${optionlist}")
    endif()
endfunction()

### ------------------------------ ###
### ---------- C FLAGS ----------- ###
### ------------------------------ ###
# This macro checks the flags in the flaglist, and adds the first one that
# is accepted by the compiler to the compile options for all subsequent targets
# (via add_compile_options)
function(AddOptionalCFlag flaglist)
    AddFlagInternal("C" "${flaglist}" FAILURE_IS_IGNORED)
endfunction()

# Synonym of AddOptionalCFlag
function(AddCFlag flaglist)
    AddFlagInternal("C" "${flaglist}" FAILURE_IS_IGNORED)
endfunction()

# This macro checks the flags in the flaglist, and adds the first one that
# is accepted by the compiler to the compile options for all subsequent targets
# (via add_compile_options). If none are supported, it fails the configuration.
function(AddRequiredCFlag flaglist)
    AddFlagInternal("C" "${flaglist}" SUCCESS_CHECK)
    if(NOT SUCCESS_CHECK)
        string(REPLACE ";" "\n\t" optionlist "${flaglist}")
        cmessage(FATAL_ERROR "None of the desired flag alternatives are supported by the compiler:\n\t${optionlist}")
    endif()
endfunction()

### ------------------------------ ###
### ------- Both FLAGS ----------- ###
### ------------------------------ ###
function(AddOptionalFlag flaglist)
    AddFlagInternal("C" "${flaglist}" FAILURE_IS_IGNORED)
    AddFlagInternal("CXX" "${flaglist}" FAILURE_IS_IGNORED)
endfunction()

# Synonym of AddOptionalCFlag
function(AddFlag flaglist)
    AddFlagInternal("C" "${flaglist}" FAILURE_IS_IGNORED)
    AddFlagInternal("CXX" "${flaglist}" FAILURE_IS_IGNORED)
endfunction()

# This macro checks the flags in the flaglist, and adds the first one that
# is accepted by the compiler to the compile options for all subsequent targets
# (via add_compile_options). If none are supported, it fails the configuration.
function(AddRequiredFlag flaglist)
    AddFlagInternal("C" "${flaglist}" SUCCESS_CHECK)
    if(NOT SUCCESS_CHECK)
        string(REPLACE ";" "\n\t" optionlist "${flaglist}")
        cmessage(FATAL_ERROR "None of the desired flag alternatives are supported by the compiler:\n\t${optionlist}")
    endif()

    AddFlagInternal("CXX" "${flaglist}" SUCCESS_CHECK)
    if(NOT SUCCESS_CHECK)
        string(REPLACE ";" "\n\t" optionlist "${flaglist}")
        cmessage(FATAL_ERROR "None of the desired flag alternatives are supported by the compiler:\n\t${optionlist}")
    endif()
endfunction()

### ------------------------------ ###
### ----- Linker FLAGS ----------- ###
### ------------------------------ ###
# This is a simple utility macro to reduce code duplication
macro(AppendLinkerFlagToCMakeVars flag)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${flag}" PARENT_SCOPE)
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${flag}" PARENT_SCOPE)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${flag}" PARENT_SCOPE)
endmacro()

# CMAKE_STATIC_LINKER_FLAGS are passed to "ar" and not "ld"
# (which uses very different command line options)
# http://cmake.3232098.n2.nabble.com/CMake-incorrectly-passes-linker-flags-to-ar-td7592436.html
macro(AppendStaticLinkerFlagToCMakeVars flag)
    set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} ${flag}" PARENT_SCOPE)
endmacro()

macro(AddLinkerFlagInternal flaglist success_var)
    set(${success_var} OFF)
    foreach(flag_under_test IN LISTS flaglist)
        # First, we convert it to a flag that will get passed to the linker...
        string(REPLACE " " "," linker_flag "${flag_under_test}")
        set(linker_flag "-Wl,${linker_flag}")
        string(MAKE_C_IDENTIFIER "${flag_under_test}" safe_flag_name)
        # Now, we test:
        DoesLinkerSupportFlag("${linker_flag}" ${safe_flag_name})
        if(LINKER_SUPPORTS_${safe_flag_name})
            message(STATUS "Adding Linker flag:       ${flag_under_test}")
            AppendLinkerFlagToCMakeVars("${linker_flag}")
            set(${success_var} ON)
            break()
        endif()
        # This fallback is a result of weird compiler/linker behavior
        # discovered when checking the behavior of --coverage
        DoesLinkerSupportFlag("${flag_under_test}" ${safe_flag_name}_FALLBACK)
        if(LINKER_SUPPORTS_${safe_flag_name}_FALLBACK)
            message(STATUS "Adding Linker flag:       ${flag_under_test}")
            AppendLinkerFlagToCMakeVars("${flag_under_test}")
            set(${success_var} ON)
            break()
        endif()
    endforeach()
endmacro()

# This function checks the flags in the flaglist, and adds the first one that is
# accepted by the compiler to the CMAKE_*_LINKER_FLAGS variables (which
# apply to all C++ targets)
function(AddOptionalLinkerFlag flaglist)
    AddLinkerFlagInternal("${flaglist}" SUCCESS_IS_IGNORED)
endfunction()

# Synonym of AddOptionalLinkerFlag
function(AddLinkerFlag flaglist)
    AddLinkerFlagInternal("${flaglist}" SUCCESS_IS_IGNORED)
endfunction()

# Add static linker flag to "ar"
function(AddStaticLinkerFlag flaglist)
    AppendStaticLinkerFlagToCMakeVars("${flaglist}" SUCCESS_IS_IGNORED)
endfunction()

# This function checks the flags in the flaglist, and adds the first one that is
# accepted by the compiler to the CMAKE_*_LINKER_FLAGS variables (which
# apply to all C++ targets). If none are supported, it fails the configuration.
function(AddRequiredLinkerFlag flaglist)
    AddLinkerFlagInternal("${flaglist}" SUCCESS_CHECK)
    if(NOT SUCCESS_CHECK)
        cmessage(FATAL_ERROR "None of the desired linker flags are supported by the compiler: ${flaglist}")
    endif()
endfunction()

# This function checks the flags in the flaglist, and adds the first one that is
# accepted by the compiler to the target using set_target_properties()
function(AddCXXFlagToTarget target flaglist)
    AddFlagToTarget("CXX" "${target}" "${flaglist}")
endfunction()

function(AddCFlagToTarget target flaglist)
    AddFlagToTarget("C" "${target}" "${flaglist}")
endfunction()

function(AddFlagToTarget lang target flaglist)
    foreach(flag_under_test ${flaglist})
        if("${flag_under_test}" MATCHES "-D.*")
            target_compile_definitions(${target} PRIVATE ${flag_under_test})
            break()
        endif()
        DoesCompilerSupportFlag(${lang} ${flag_under_test} safe_flag_name)
        if(${${lang}_COMPILER_SUPPORTS_${safe_flag_name}})
            message(STATUS "Adding ${lang} Compiler flag to ${target}: ${flag_under_test}")
            target_compile_options(${target} PRIVATE ${flag_under_test})
            break()
        endif()
    endforeach()
endfunction()

# vim: sw=4 expandtab
