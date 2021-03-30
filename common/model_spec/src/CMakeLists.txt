file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# shared library
add_library(${PROJECT_NAME} SHARED ${srcs})
target_link_libraries(${PROJECT_NAME} LINK_PUBLIC uni)

# static library
add_library(${PROJECT_NAME}_static STATIC ${srcs})

set_target_properties(${PROJECT_NAME}_static PROPERTIES OUTPUT_NAME "${PROJECT_NAME}")
set_target_properties(${PROJECT_NAME} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(${PROJECT_NAME}_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
