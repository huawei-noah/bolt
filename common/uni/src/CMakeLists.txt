file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

include_directories(${OPENCL_INCLUDE_DIR})

# shared library
add_library(${PROJECT_NAME} SHARED ${srcs})

# static library
add_library(${PROJECT_NAME}_static STATIC ${srcs})
link_securec(${PROJECT_NAME})

set_target_properties(${PROJECT_NAME}_static PROPERTIES OUTPUT_NAME "${PROJECT_NAME}")
set_target_properties(${PROJECT_NAME} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(${PROJECT_NAME}_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION lib
        LIBRARY DESTINATION lib)
