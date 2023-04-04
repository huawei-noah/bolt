set(srcs "model_spec.cpp;model_deserialize.cpp")
if (NOT USE_LITE OR USE_CAFFE OR USE_ONNX OR USE_TFLITE OR USE_MINDSPORE)
    set(srcs "${srcs};model_serialize.cpp;model_print.cpp;model_common.cpp")
endif ()

# shared library
add_library(${PROJECT_NAME} SHARED ${srcs})
target_link_libraries(${PROJECT_NAME} uni)
#link_uni_shared(${PROJECT_NAME})

# static library
add_library(${PROJECT_NAME}_static STATIC ${srcs})

set_target_properties(${PROJECT_NAME}_static PROPERTIES OUTPUT_NAME "${PROJECT_NAME}")
set_target_properties(${PROJECT_NAME} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(${PROJECT_NAME}_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION lib
        LIBRARY DESTINATION lib)
