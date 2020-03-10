cmake_minimum_required(VERSION 3.2)

file(GLOB BOLT_CONFIGURE_FILE $ENV{BOLT_ROOT}/bolt.cmake ${BOLT_ROOT}/bolt.cmake)
if (BOLT_CONFIGURE_FILE)
    include(${BOLT_CONFIGURE_FILE})
else (BOLT_CONFIGURE_FILE)
    message(FATAL_ERROR "
FATAL: can not find bolt.cmake in <BOLT_ROOT> directory,
       please set shell or cmake environment variable BOLT_ROOT.
    ")
endif (BOLT_CONFIGURE_FILE)

project(model-tools)

set_policy()

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${BOLT_ROOT}/cmakes")
find_package(Uni)
find_package(ModelTools)

set_project_install_directory()

set_c_cxx_flags()

add_subdirectory(src)
if (USE_CAFFE)
    add_subdirectory(src/caffe)
endif(USE_CAFFE)
if (USE_ONNX)
    add_subdirectory(src/onnx)
endif(USE_ONNX)
if (USE_TFLITE)
    add_subdirectory(src/tflite)
endif(USE_TFLITE)
