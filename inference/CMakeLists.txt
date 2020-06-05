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

project(inference)

set_policy()

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${BOLT_ROOT}/cmakes")
find_package(Uni)
find_package(ModelTools)
find_package(Image)
find_package(TensorComputing)
if(USE_MALI)
    find_package(Gcl)
endif(USE_MALI)
if(BUILD_TEST)
    find_package(jpeg)
endif(BUILD_TEST)

set_project_install_directory()

set_c_cxx_flags()

add_subdirectory(src)
