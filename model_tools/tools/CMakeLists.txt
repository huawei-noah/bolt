set_test_c_cxx_flags()

model_tools_test(X2bolt "X2bolt/X2bolt.cpp")
install(TARGETS X2bolt
        RUNTIME DESTINATION tools)
engine_test(post_training_quantization "quantization/post_training_quantization.cpp")
link_model_tools(post_training_quantization)
install(TARGETS post_training_quantization
        RUNTIME DESTINATION tools)
