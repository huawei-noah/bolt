file(GLOB srcs *.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${TFLITE_INCLUDE_DIR})

include_directories(../)

include_directories(${Protobuf_INCLUDE_DIR})

# shared library
add_library(${PROJECT_NAME}_tflite SHARED ${srcs})
target_link_libraries (${PROJECT_NAME}_tflite LINK_PUBLIC model_spec uni ${Protobuf_SHARED_LIBRARY})

# static library
add_library(${PROJECT_NAME}_tflite_static STATIC ${srcs})

set_target_properties(${PROJECT_NAME}_tflite_static PROPERTIES OUTPUT_NAME "${PROJECT_NAME}_tflite")
set_target_properties(${PROJECT_NAME}_tflite PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(${PROJECT_NAME}_tflite_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
install(TARGETS ${PROJECT_NAME}_tflite
        RUNTIME DESTINATION lib
        LIBRARY DESTINATION lib)
