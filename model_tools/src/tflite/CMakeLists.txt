file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB commonsrcs ${CMAKE_CURRENT_SOURCE_DIR}/../model_*.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${TFLITE_INCLUDE_DIR})

include_directories(../)

# shared library
add_library(${PROJECT_NAME}_tflite SHARED ${srcs} ${commonsrcs})
target_link_libraries (${PROJECT_NAME}_tflite LINK_PUBLIC uni)

# static library
add_library(${PROJECT_NAME}_tflite_static STATIC ${srcs})

set_target_properties(${PROJECT_NAME}_tflite_static PROPERTIES OUTPUT_NAME "${PROJECT_NAME}_tflite")
set_target_properties(${PROJECT_NAME}_tflite PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(${PROJECT_NAME}_tflite_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
install(TARGETS ${PROJECT_NAME}_tflite ${PROJECT_NAME}_tflite_static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
