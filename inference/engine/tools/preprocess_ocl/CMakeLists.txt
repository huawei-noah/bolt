cmake_minimum_required(VERSION 3.2)

project(kernelbin)

include_directories(${PROJECT_SOURCE_DIR}/tmp/include)
include_directories(${PROJECT_SOURCE_DIR}/tmp/extern)

file(GLOB_RECURSE kernel_bin_src_list ${PROJECT_SOURCE_DIR}/tmp/cpp/*.cpp)

add_library(${PROJECT_NAME} SHARED ${kernel_bin_src_list})
