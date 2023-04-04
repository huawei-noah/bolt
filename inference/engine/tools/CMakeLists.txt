cmake_minimum_required(VERSION 3.2)

set_test_c_cxx_flags()

if (BUILD_EXAMPLE)
    engine_test(common_algo_search ./common_algo_search/common_algo_search.cpp)
    install(TARGETS common_algo_search
            RUNTIME DESTINATION tools)
endif ()
if (USE_GPU)
    engine_test(preprocess_ocl ./preprocess_ocl/preprocess_ocl.cpp)
    engine_test(update_ocl ./preprocess_ocl/update_ocl.cpp)
    install(TARGETS preprocess_ocl update_ocl
            RUNTIME DESTINATION tools/preprocess_ocl)
    install(FILES ./preprocess_ocl/preprocess_ocl.sh
            ../../../scripts/setup_compiler.sh
            PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ
            DESTINATION tools/preprocess_ocl)
    install(FILES ./preprocess_ocl/CMakeLists.txt
            DESTINATION tools/preprocess_ocl)
    install(FILES
            ../../../common/gcl/include/gcl_kernel_binmap.h
            ../../../common/uni/include/file.h
            ../../../common/uni/include/string_functions.h
            ../../../common/uni/include/secure_c_wrapper.h
            ../../../common/uni/include/error.h
            DESTINATION tools/preprocess_ocl/include)
endif (USE_GPU)
#if (USE_TRAINING)
#    train_test(model_finetuner ./model_finetuner/model_finetuner.cpp)
#    install(TARGETS model_finetuner
#            RUNTIME DESTINATION tools)
#endif (USE_TRAINING)
