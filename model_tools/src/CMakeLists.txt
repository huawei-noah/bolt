file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

include_directories(${JSONCPP_INCLUDE_DIR})

# shared library 
add_library(${PROJECT_NAME} SHARED ${srcs})

# static library 
add_library(${PROJECT_NAME}_static STATIC ${srcs})

if (USE_CAFFE)
    add_subdirectory(caffe)
    target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_caffe)
endif(USE_CAFFE)
if (USE_ONNX)
    add_subdirectory(onnx)
    target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_onnx)
endif(USE_ONNX)
if (USE_TFLITE)
    add_subdirectory(tflite)
    target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_tflite)
endif(USE_TFLITE)
if (USE_TENSORFLOW)
    add_subdirectory(tensorflow)
    target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_tensorflow)
endif(USE_TENSORFLOW)
if (USE_MINDSPORE)
    add_subdirectory(mindspore)
    target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_mindspore)
endif(USE_MINDSPORE)
link_uni(${PROJECT_NAME})

set_target_properties(${PROJECT_NAME}_static PROPERTIES OUTPUT_NAME "${PROJECT_NAME}")
set_target_properties(${PROJECT_NAME} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(${PROJECT_NAME}_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION lib
        LIBRARY DESTINATION lib)
