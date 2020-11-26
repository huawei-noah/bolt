file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB commonsrcs ${CMAKE_CURRENT_SOURCE_DIR}/../model_*.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${Protobuf_INCLUDE_DIR})

set(Protobuf_IMPORT_DIRS ${BOLT_ROOT}/third_party/proto)
protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS ${BOLT_ROOT}/third_party/proto/onnx.proto)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(../)

# shared library
add_library(${PROJECT_NAME}_onnx SHARED ${srcs} ${ONNX_PROTO_HDRS} ${ONNX_PROTO_SRCS} ${commonsrcs})
if (USE_IOS_CLANG)
    target_link_libraries(${PROJECT_NAME}_onnx LINK_PUBLIC uni ${Protobuf_LIBRARY})
endif (USE_IOS_CLANG)

# static library
add_library(${PROJECT_NAME}_onnx_static STATIC ${srcs} ${ONNX_PROTO_HDRS} ${ONNX_PROTO_SRCS})

set_target_properties(${PROJECT_NAME}_onnx_static PROPERTIES OUTPUT_NAME "${PROJECT_NAME}_onnx")
set_target_properties(${PROJECT_NAME}_onnx PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(${PROJECT_NAME}_onnx_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
install(TARGETS ${PROJECT_NAME}_onnx ${PROJECT_NAME}_onnx_static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
