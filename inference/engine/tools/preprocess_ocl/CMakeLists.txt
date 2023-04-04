cmake_minimum_required(VERSION 3.2)

project(kernelbin)

include_directories(${CMAKE_SOURCE_DIR}/tmp/include)
if(EXISTS ${CMAKE_SOURCE_DIR}/include)
    include_directories(${CMAKE_SOURCE_DIR}/include)
else ()
    include_directories(${CMAKE_SOURCE_DIR}/../../../../common/gcl/include)
    include_directories(${CMAKE_SOURCE_DIR}/../../../../common/uni/include)
endif ()

file(GLOB_RECURSE srcs ${CMAKE_SOURCE_DIR}/tmp/cpp/*.cpp)
string(REGEX REPLACE ".cpp" "" v ${srcs})
string(REGEX REPLACE ".*/" "" lib ${v})

set(COMMON_FLAGS "-O3 -fPIE -fPIC")
set(COMMON_FLAGS "${COMMON_FLAGS} -Wno-unused-command-line-argument -Wno-unused-parameter")
set(COMMON_FLAGS "${COMMON_FLAGS} -Wno-unused-result -Wno-deprecated-declarations -Wno-unused-variable")
if ((NOT CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 8.1.0) OR (NOT WIN32))
    set(COMMON_FLAGS "${COMMON_FLAGS} -fstack-protector-all -fstack-protector-strong")
endif()
set(COMMON_FLAGS "${COMMON_FLAGS} -static-libstdc++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS}")

add_library(${lib} SHARED ${srcs})
install(TARGETS ${lib}
        LIBRARY DESTINATION ./
        RUNTIME DESTINATION ./
        ARCHIVE DESTINATION ./)
