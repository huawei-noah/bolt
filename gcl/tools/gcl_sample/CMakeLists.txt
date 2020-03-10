cmake_minimum_required(VERSION 3.4.1)

file(GLOB BOLT_CONFIGURE_FILE $ENV{BOLT_ROOT}/bolt.cmake ${BOLT_ROOT}/bolt.cmake)
if (BOLT_CONFIGURE_FILE)
    include(${BOLT_CONFIGURE_FILE})
else (BOLT_CONFIGURE_FILE)
    message(FATAL_ERROR "
FATAL: can not find bolt.cmake in <BOLT_ROOT> directory,
       please set shell or cmake environment variable BOLT_ROOT.
    ")
endif (BOLT_CONFIGURE_FILE)

project(sample)

set_policy()

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${BOLT_ROOT}/cmakes")
find_package(Gcl)
find_package(Uni)

set_c_cxx_flags()

set_test_c_cxx_flags()

add_executable(sample sample.cpp)
target_link_libraries(sample ${KERNELBIN_LIBRARIES} ${OPENCL_LIBRARIES})
