set(srcs "${CMAKE_CURRENT_SOURCE_DIR}/model_tools.cpp")

if(BUILD_TEST OR USE_CAFFE OR USE_ONNX OR USE_TFLITE OR USE_TENSORFLOW)
    set(srcs "${srcs};${CMAKE_CURRENT_SOURCE_DIR}/data_type_converter.cpp;${CMAKE_CURRENT_SOURCE_DIR}/online_conversion.cpp;${CMAKE_CURRENT_SOURCE_DIR}/model_quantization.cpp;")
endif(BUILD_TEST OR USE_CAFFE OR USE_ONNX OR USE_TFLITE OR USE_TENSORFLOW)

# shared library 
add_library(${PROJECT_NAME} SHARED ${srcs})

# static library 
add_library(${PROJECT_NAME}_static STATIC ${srcs})

if (USE_CAFFE)
    add_subdirectory(caffe)
    target_link_libraries (${PROJECT_NAME} LINK_PUBLIC ${PROJECT_NAME}_caffe)
endif(USE_CAFFE)
if (USE_ONNX)
    add_subdirectory(onnx)
    target_link_libraries (${PROJECT_NAME} LINK_PUBLIC ${PROJECT_NAME}_onnx)
endif(USE_ONNX)
if (USE_TFLITE)
    add_subdirectory(tflite)
    target_link_libraries (${PROJECT_NAME} LINK_PUBLIC ${PROJECT_NAME}_tflite)
endif(USE_TFLITE)
if (USE_TENSORFLOW)
    add_subdirectory(tensorflow)
    target_link_libraries (${PROJECT_NAME} LINK_PUBLIC ${PROJECT_NAME}_tensorflow)
endif(USE_TENSORFLOW)
target_link_libraries (${PROJECT_NAME} LINK_PUBLIC uni)

set_target_properties(${PROJECT_NAME}_static PROPERTIES OUTPUT_NAME "${PROJECT_NAME}")
set_target_properties(${PROJECT_NAME} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(${PROJECT_NAME}_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
