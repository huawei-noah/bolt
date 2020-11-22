cmake_minimum_required(VERSION 3.4.1)

file(GLOB BOLT_CONFIGURE_FILE $ENV{BOLT_ROOT}/common/cmakes/bolt.cmake ${BOLT_ROOT}/common/cmakes/bolt.cmake)
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

find_package(Gcl)

set_c_cxx_flags()
set_test_c_cxx_flags()

add_executable(sample sample.cpp)
target_link_libraries(sample ${KERNELSOURCE_LIBRARIES} ${OPENCL_LIBRARIES} log -Wl,-allow-shlib-undefined, -static-libstdc++)
