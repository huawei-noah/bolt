cmake_minimum_required(VERSION 3.2)

set_test_c_cxx_flags()

if (BUILD_TEST)
    engine_test(common_algo_search ./common_algo_search/common_algo_search.cpp)
    install(TARGETS common_algo_search
            RUNTIME DESTINATION tools)
endif (BUILD_TEST)
if (BUILD_TEST AND USE_INT8)
    engine_test(ptq_calibration ./ptq_calibration/ptq_calibration.cpp)
    install(TARGETS ptq_calibration
            RUNTIME DESTINATION tools)
endif (BUILD_TEST AND USE_INT8)
if (USE_MALI)
    engine_test(preprocess_ocl ./preprocess_ocl/preprocess_ocl.cpp)
    install(TARGETS preprocess_ocl
            RUNTIME DESTINATION tools)
endif (USE_MALI)
if (USE_TRAINING)
    train_test(model_finetuner ./model_finetuner/model_finetuner.cpp)
    TARGET_LINK_LIBRARIES(model_finetuner RaulLib)
    install(TARGETS model_finetuner
            RUNTIME DESTINATION tools)
endif (USE_TRAINING)
