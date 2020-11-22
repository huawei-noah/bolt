cmake_minimum_required(VERSION 3.2)

project(kernelbin)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/extern)

file(GLOB_RECURSE kernel_bin_src_list cpp/*.cpp)

add_library(${PROJECT_NAME} SHARED ${kernel_bin_src_list})
