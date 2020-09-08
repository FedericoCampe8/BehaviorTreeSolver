# This allows us to use the "officially recommended" method of downloading
# googletest with git.

# This macro exists to assist the add_{gtest|gmock}_to_target macros. The
# problem is that GoogleTest (as of 1.8.x) does some unfortunate things that
# are not fully compatible with -Wall. Disabling these warnings for targets
# that use it is harmless if you don't use -Wall and avoids build errors if you
# do (since we enforce -Werror)
macro(add_gtest_warning_flags_to_target target)
  # GoogleTest makes comparisons between signed and unsigned integer expressions
  AddCXXFlagToTarget(${target} "-Wno-sign-compare")

  # GoogleTest uses C99-style macros, even though they're not officially part of C++
  AddCXXFlagToTarget(${target} "-Wno-c99-extensions")
  AddCXXFlagToTarget(${target} "-Wno-variadic-macros")

  # GoogleTest uses commas at the end of enumerator lists, even though it's a C++11 extensions
  AddCXXFlagToTarget(${target} "-Wno-c++11-extensions")
endmacro()
macro(add_gmock_warning_flags_to_target target)
  # Ignore Gmock's limitations
  AddCXXFlagToTarget(${target} "-Wno-inconsistent-missing-override")
endmacro()

include(OfficialGtestSetup)
