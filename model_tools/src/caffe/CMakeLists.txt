file(GLOB srcs *.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${Protobuf_INCLUDE_DIR})

set(Protobuf_IMPORT_DIRS ${BOLT_ROOT}/third_party/proto)
protobuf_generate_cpp(CAFFE_PROTO_SRCS CAFFE_PROTO_HDRS ${BOLT_ROOT}/third_party/proto/caffe.proto)

protobuf_generate_python(CAFFE_PROTO_PYTHON_SRCS ${BOLT_ROOT}/third_party/proto/caffe.proto)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(../)

add_custom_target(caffe_pb2.py ALL
                  DEPENDS ${CAFFE_PROTO_PYTHON_SRCS}
                  COMMAND ${CMAKE_COMMAND} -E copy ${CAFFE_PROTO_PYTHON_SRCS} ${PROJECT_SOURCE_DIR}/tools/tensorflow2caffe/Caffe)

# shared library 
add_library(${PROJECT_NAME}_caffe SHARED ${srcs} ${CAFFE_PROTO_HDRS} ${CAFFE_PROTO_SRCS})
target_link_libraries(${PROJECT_NAME}_caffe LINK_PUBLIC model_spec uni ${Protobuf_SHARED_LIBRARY})
# static library 
add_library(${PROJECT_NAME}_caffe_static STATIC ${srcs} ${CAFFE_PROTO_HDRS} ${CAFFE_PROTO_SRCS})

set_target_properties(${PROJECT_NAME}_caffe_static PROPERTIES OUTPUT_NAME "${PROJECT_NAME}_caffe")
set_target_properties(${PROJECT_NAME}_caffe PROPERTIES CLEAN_DIRECT_OUTPUT 1) 
set_target_properties(${PROJECT_NAME}_caffe_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
install(TARGETS ${PROJECT_NAME}_caffe ${PROJECT_NAME}_caffe_static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
