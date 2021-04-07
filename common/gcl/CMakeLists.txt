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

project(gcl)

set_c_cxx_flags()

include_gcl()

add_subdirectory(src)
add_subdirectory(tools/kernel_source_compile)
add_subdirectory(tools/device_info)
add_subdirectory(tools/gcl_sample)
