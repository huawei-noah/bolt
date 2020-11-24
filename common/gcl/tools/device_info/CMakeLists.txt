cmake_minimum_required(VERSION 3.2)

file(GLOB BOLT_CONFIGURE_FILE $ENV{BOLT_ROOT}/common/cmakes/bolt.cmake ${BOLT_ROOT}/common/cmakes/bolt.cmake)
if (BOLT_CONFIGURE_FILE)
    set(USE_LLVM_CLANG ON)
    set(USE_GNU_GCC OFF)
    set(USE_MALI ON)
    set(USE_DYNAMIC_LIBRARY OFF)
    include(${BOLT_CONFIGURE_FILE})
else (BOLT_CONFIGURE_FILE)
    message(FATAL_ERROR "
FATAL: can not find bolt.cmake in <BOLT_ROOT> directory,
       please set shell or cmake environment variable BOLT_ROOT.
    ")
endif (BOLT_CONFIGURE_FILE)

project(gclinfo)

set_policy()

find_package(Gcl)

set_c_cxx_flags()

set_test_c_cxx_flags()

add_executable(gcl_info clinfo.cpp)
target_link_libraries(gcl_info ${OPENCL_LIBRARIES} log -Wl,-allow-shlib-undefined, -static-libstdc++)
