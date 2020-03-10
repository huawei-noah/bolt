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

project(KERNELBIN)

set_policy()

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${BOLT_ROOT}/cmakes")
find_package(Gcl)
find_package(Uni)
include_directories(${PROJECT_SOURCE_DIR}/include)
set_project_install_directory()

set_c_cxx_flags()

execute_process(
    COMMAND bash buildKernelLib.sh
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

file(GLOB_RECURSE kernel_src_list "src/*.cpp")
ADD_LIBRARY(kernelbin SHARED ${kernel_src_list})
ADD_LIBRARY(kernelbin_static STATIC ${kernel_src_list})
SET_TARGET_PROPERTIES(kernelbin_static PROPERTIES OUTPUT_NAME "kernelbin")
SET_TARGET_PROPERTIES(kernelbin PROPERTIES CLEAN_DIRECT_OUTPUT 1)
SET_TARGET_PROPERTIES(kernelbin_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
