file(GLOB srcs *.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${JSONCPP_INCLUDE_DIR})

include_directories(../)

# shared library
add_library(${PROJECT_NAME}_tensorflow SHARED ${srcs})
target_link_libraries(${PROJECT_NAME}_tensorflow LINK_PUBLIC model_spec ${JSONCPP_SHARED_LIBRARY} uni)

# static library
add_library(${PROJECT_NAME}_tensorflow_static STATIC ${srcs})

set_target_properties(${PROJECT_NAME}_tensorflow_static PROPERTIES OUTPUT_NAME "${PROJECT_NAME}_tensorflow")
set_target_properties(${PROJECT_NAME}_tensorflow PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(${PROJECT_NAME}_tensorflow_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
install(TARGETS ${PROJECT_NAME}_tensorflow ${PROJECT_NAME}_tensorflow_static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
