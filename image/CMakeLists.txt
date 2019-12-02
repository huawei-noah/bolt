cmake_minimum_required(VERSION 3.2)

file(GLOB BOLT_CONFIGURE_FILE $ENV{BOLT_ROOT}/bolt.cmake ${BOLT_ROOT}/bolt.cmake)
if (BOLT_CONFIGURE_FILE)
    include(${BOLT_CONFIGURE_FILE})
endif (BOLT_CONFIGURE_FILE)


project(image)


set_policy()
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../uni/cmakes")
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmakes")
find_package(Uni)
find_package(Image)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra")
# add CImg jpeg and png support
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Dcimg_use_png -Dcimg_use_jpeg")

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

if (USE_DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif(USE_DEBUG)

add_subdirectory(src)
add_subdirectory(tests)
