file(GLOB srcs *.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${JSONCPP_INCLUDE_DIR})

include_directories(../)

# shared library
add_library(${PROJECT_NAME}_tensorflow SHARED ${srcs})
target_link_libraries(${PROJECT_NAME}_tensorflow ${JSONCPP_SHARED_LIBRARY})
link_model_spec(${PROJECT_NAME}_tensorflow)

# static library
add_library(${PROJECT_NAME}_tensorflow_static STATIC ${srcs})

set_target_properties(${PROJECT_NAME}_tensorflow_static PROPERTIES OUTPUT_NAME "${PROJECT_NAME}_tensorflow")
set_target_properties(${PROJECT_NAME}_tensorflow PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(${PROJECT_NAME}_tensorflow_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
install(TARGETS ${PROJECT_NAME}_tensorflow
        RUNTIME DESTINATION lib
        LIBRARY DESTINATION lib)
