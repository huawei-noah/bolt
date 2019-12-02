cmake_minimum_required(VERSION 3.2)

file(GLOB BOLT_CONFIGURE_FILE $ENV{BOLT_ROOT}/bolt.cmake ${BOLT_ROOT}/bolt.cmake)
if (BOLT_CONFIGURE_FILE)
    include(${BOLT_CONFIGURE_FILE})
endif (BOLT_CONFIGURE_FILE)


project(model-tools)


set_policy()
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../uni/cmakes")
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmakes")
find_package(Uni)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -O2")

if (USE_DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif(USE_DEBUG)

if (USE_CAFFE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_CAFFE_MODEL")
    find_package(ModelTools)
endif()

if (USE_ONNX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_ONNX_MODEL")
    find_package(ModelTools)
endif()

add_subdirectory(src)
add_subdirectory(src/caffe)
add_subdirectory(src/onnx)
add_subdirectory(tests)
