set_test_c_cxx_flags()

if(BUILD_TEST OR USE_CAFFE OR USE_ONNX OR USE_TFLITE OR USE_TENSORFLOW)
model_tools_test(X2bolt "X2bolt/X2bolt.cpp")
install(TARGETS X2bolt
        RUNTIME DESTINATION tools)
model_tools_test(post_training_quantization "quantization/post_training_quantization.cpp")
install(TARGETS post_training_quantization
        RUNTIME DESTINATION tools)
endif(BUILD_TEST OR USE_CAFFE OR USE_ONNX OR USE_TFLITE OR USE_TENSORFLOW)
