file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

include_directories(${PROJECT_SOURCE_DIR}/include)

# shared library 
add_library(${PROJECT_NAME} SHARED ${srcs})

set_target_properties(${PROJECT_NAME} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
