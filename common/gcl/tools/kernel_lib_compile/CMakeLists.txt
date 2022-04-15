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

project(KERNELBIN)

set_policy()

find_package(Gcl)
include_directories(${PROJECT_SOURCE_DIR}/include)
set_project_install_directory()

set_c_cxx_flags()

execute_process(
    COMMAND bash buildKernelLib.sh
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

file(GLOB_RECURSE kernel_src_list "src/*.cpp")
add_library(kernelbin SHARED ${kernel_src_list})
add_library(kernelbin_static STATIC ${kernel_src_list})
set_target_properties(kernelbin_static PROPERTIES OUTPUT_NAME "kernelbin")
set_target_properties(kernelbin PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(kernelbin_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
