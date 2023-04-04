file(GLOB srcs *.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${Protobuf_INCLUDE_DIR})

set(Protobuf_IMPORT_DIRS ${BOLT_ROOT}/third_party/proto)
protobuf_generate_cpp(MINDSPORE_PROTO_SRCS MINDSPORE_PROTO_HDRS ${BOLT_ROOT}/third_party/proto/mind_ir.proto)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(../)

# shared library
add_library(${PROJECT_NAME}_mindspore SHARED ${srcs} ${MINDSPORE_PROTO_HDRS} ${MINDSPORE_PROTO_SRCS})

target_link_libraries(${PROJECT_NAME}_mindspore ${Protobuf_SHARED_LIBRARY})
link_model_spec(${PROJECT_NAME}_mindspore)

# static library
add_library(${PROJECT_NAME}_mindspore_static STATIC ${srcs} ${MINDSPORE_PROTO_HDRS} ${MINDSPORE_PROTO_SRCS})

set_target_properties(${PROJECT_NAME}_mindspore_static PROPERTIES OUTPUT_NAME "${PROJECT_NAME}_mindspore")
set_target_properties(${PROJECT_NAME}_mindspore PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(${PROJECT_NAME}_mindspore_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
install(TARGETS ${PROJECT_NAME}_mindspore
        RUNTIME DESTINATION lib
        LIBRARY DESTINATION lib)
