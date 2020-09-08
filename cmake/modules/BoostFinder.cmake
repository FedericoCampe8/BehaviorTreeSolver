macro(find_boost)
  set(boost_components ${ARGN}) # This allows us to make arguments to this macro optional
  list(LENGTH boost_components num_components)
  if (${num_components} GREATER 0)
    find_package(Boost 1.70 REQUIRED COMPONENTS ${boost_components})
  else()
    find_package(Boost 1.70 REQUIRED)
  endif()
  # Boost-1.70 uses unused local typedefs *repeatedly*
  AddCXXFlag("-Wno-unused-local-typedefs")
endmacro()
