cmake_minimum_required(VERSION 3.2)

file(GLOB BOLT_CONFIGURE_FILE $ENV{BOLT_ROOT}/common/cmakes/bolt.cmake ${BOLT_ROOT}/common/cmakes/bolt.cmake)
if (BOLT_CONFIGURE_FILE)
    include(${BOLT_CONFIGURE_FILE})
else (BOLT_CONFIGURE_FILE)
    message(FATAL_ERROR "
FATAL: can not find bolt.cmake in <BOLT_ROOT>/common/cmakes directory,
       please set shell or cmake environment variable BOLT_ROOT.
    ")
endif (BOLT_CONFIGURE_FILE)

project(gclBinary)

set_policy()

find_package(Gcl)

set_c_cxx_flags()

set_test_c_cxx_flags()

add_executable(gcl_binary clbinary.cpp)
target_link_libraries(gcl_binary ${OPENCL_LIBRARIES} -Wl,-allow-shlib-undefined, -static-libstdc++)
