file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# shared library 
add_library(${PROJECT_NAME} SHARED ${srcs})

# static library 
add_library(${PROJECT_NAME}_static STATIC ${srcs})

target_link_libraries(${PROJECT_NAME} LINK_PUBLIC tensor image model_spec)
if (BUILD_TEST)
    target_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${JPEG_SHARED_LIBRARY})
endif (BUILD_TEST)

set_target_properties(${PROJECT_NAME}_static PROPERTIES OUTPUT_NAME "${PROJECT_NAME}")
set_target_properties(${PROJECT_NAME} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(${PROJECT_NAME}_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
