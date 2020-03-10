file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

protobuf_generate_cpp(CAFFE_PROTO_SRCS CAFFE_PROTO_HDRS caffe.proto)

protobuf_generate_python(CAFFE_PROTO_PYTHON_SRCS caffe.proto)

include_directories(${PROTOBUF_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
include_directories(../)

add_custom_target(caffe_pb2.py ALL
                  DEPENDS ${CAFFE_PROTO_PYTHON_SRCS}
                  COMMAND ${CMAKE_COMMAND} -E copy ${CAFFE_PROTO_PYTHON_SRCS} ${PROJECT_SOURCE_DIR}/tools/tensorflow2caffe/Caffe)

# shared library 
ADD_LIBRARY(${PROJECT_NAME}_caffe SHARED ${srcs} ${CAFFE_PROTO_HDRS} ${CAFFE_PROTO_SRCS}) 

# static library 
ADD_LIBRARY(${PROJECT_NAME}_caffe_static STATIC ${srcs} ${CAFFE_PROTO_HDRS} ${CAFFE_PROTO_SRCS})

SET_TARGET_PROPERTIES(${PROJECT_NAME}_caffe_static PROPERTIES OUTPUT_NAME "${PROJECT_NAME}_caffe") 
SET_TARGET_PROPERTIES(${PROJECT_NAME}_caffe PROPERTIES CLEAN_DIRECT_OUTPUT 1) 
SET_TARGET_PROPERTIES(${PROJECT_NAME}_caffe_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
