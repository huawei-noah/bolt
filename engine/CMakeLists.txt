cmake_minimum_required(VERSION 3.2)

file(GLOB BOLT_CONFIGURE_FILE $ENV{BOLT_ROOT}/bolt.cmake ${BOLT_ROOT}/bolt.cmake)
if (BOLT_CONFIGURE_FILE)
    include(${BOLT_CONFIGURE_FILE})
endif (BOLT_CONFIGURE_FILE)


project(engine)


set_policy()
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../uni/cmakes")
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../model-tools/cmakes")
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../image/cmakes")
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../tensor_computing/cmakes")

find_package(Uni)
find_package(ModelTools)
find_package(Image)
find_package(TensorComputing)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra")

if (USE_DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -D_DEBUG")
endif(USE_DEBUG)

if (USE_NEON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_NEON")
endif(USE_NEON)

if (USE_INT8)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_INT8")
endif(USE_INT8)

add_subdirectory(src)
add_subdirectory(tests)
