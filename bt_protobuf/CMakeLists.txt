set(protobuf_MODULE_COMPATIBLE ON CACHE BOOL "")
set(Protobuf_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/third-party/protobuf_3_12_2")
set(Protobuf_INCLUDE_DIR_OR_TOOLS "${PROJECT_SOURCE_DIR}/include/or-tools")

set(proto_arch_dir "")

if( UNIX AND NOT APPLE )
  set(proto_arch_dir "linux")
elseif( APPLE )
  set(proto_arch_dir "mac")
endif()

set(Protobuf_LIBRARIES "${PROJECT_SOURCE_DIR}/lib/${proto_arch_dir}/protobuf_3_12_2")
set(Protobuf_PROTOC_EXECUTABLE "${PROJECT_SOURCE_DIR}/bin/${proto_arch_dir}/protobuf/protoc")

# On APPLE architecture export dyld to run protoc
if( APPLE )
  set(ENV{DYLD_LIBRARY_PATH} ${Protobuf_LIBRARIES})
endif()

# Set input/output directories
set(PROTOBUF_INPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/optilab_protobuf")
set(PROTOBUF_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/optilab_protobuf")

# find the protobuf compiler and libraries
# find_package(Protobuf REQUIRED)

# Generate the .h and .cxx files
file(GLOB PROTO_SOURCES "*.proto")
#PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${PROTO_SOURCES})

foreach(file ${PROTO_SOURCES})
  set(PROTOBUF_ARGUMENTS ${Protobuf_PROTOC_EXECUTABLE} --proto_path=${Protobuf_INCLUDE_DIR} --proto_path=${PROTOBUF_INPUT_DIRECTORY} --proto_path=${Protobuf_INCLUDE_DIR_OR_TOOLS} --python_out=${PROTOBUF_OUTPUT_DIRECTORY} --cpp_out=${PROTOBUF_OUTPUT_DIRECTORY} ${file})
  execute_process(COMMAND ${PROTOBUF_ARGUMENTS}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/optilab_protobuf
        RESULT_VARIABLE PROTOBUF_RESULT
        OUTPUT_VARIABLE PROTOBUF_OUTPUT_VARIABLE)
  if (NOT ${PROTOBUF_RESULT} EQUAL 0)
    message(FATAL_ERROR "Protobuf build failed " ${PROTOBUF_OUTPUT_VARIABLE})
  endif()
endforeach()

file(GLOB PROTOBUF_MODELS_INCLUDES "${PROTOBUF_OUTPUT_DIRECTORY}/*.pb.cc")

# Ensure that the value of this variable is transmitted to a higher scope
set(PROTO_SRCS "${PROTOBUF_MODELS_INCLUDES}" PARENT_SCOPE)

# Avoid the below
#set(PROTO_HDRS "${PROTO_HDRS}" PARENT_SCOPE)

# Add the files to the protobuf library
#add_library(OptiLabProtobuf SHARED ${PROTO_SRCS})
#install(EXPORT OptiLabProtobuf DESTINATION ${INSTALL_LIBDIR})
