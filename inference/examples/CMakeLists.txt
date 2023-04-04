cmake_minimum_required(VERSION 3.2)

file(GLOB BOLT_CONFIGURE_FILE $ENV{BOLT_ROOT}/common/cmakes/bolt.cmake ${BOLT_ROOT}/common/cmakes/bolt.cmake)
if (BOLT_CONFIGURE_FILE)
    include(${BOLT_CONFIGURE_FILE})
else (BOLT_CONFIGURE_FILE)
    message(FATAL_ERROR "
FATAL: can not find bolt.cmake in <BOLT_ROOT>/common/cmakes directory,
       please set shell or cmake environment variable BOLT_ROOT.
    ")
endif (BOLT_CONFIGURE_FILE)

project(examples)

set_c_cxx_flags()
set_test_c_cxx_flags()

include_flow()

if (BUILD_TEST)
    engine_test(benchmark benchmark/benchmark.cpp)
    install(TARGETS benchmark RUNTIME DESTINATION examples)
    if (USE_API_PYTHON)
        install(FILES benchmark/benchmark.py
                DESTINATION examples)
    endif()
endif ()
if (BUILD_EXAMPLE)
    engine_test(c_image_classification "c_api/c_image_classification.c;c_api/c_test.c")
    engine_test(c_input_method "c_api/c_input_method.c;c_api/c_test.c")
    engine_test(c_set_inout "c_api/c_set_inout.c;c_api/c_test.c")
    engine_test(bert bert/bert.cpp)
    engine_test(tinybert bert/tinybert.cpp)
    engine_test(classification image_classification/classification.cpp)
    engine_test(nmt machine_translation/nmt.cpp)
    engine_test(nmt_tsc machine_translation/nmt_tsc.cpp)
    engine_test(nmt_en-cs machine_translation/nmt_en-cs.cpp)
    engine_test(tsc_ssru machine_translation/tsc_ssru.cpp)
    engine_test(asr_rnnt automatic_speech_recognition/asr_rnnt.cpp)
    engine_test(asr_convolution_transformer automatic_speech_recognition/asr_convolution_transformer.cpp)
    engine_test(tts text_to_speech/tts.cpp)
    engine_test(vad automatic_speech_recognition/vad.cpp)
    engine_test(detection object_detection/detection.cpp)
    engine_test(tinybert_onnx bert/tinybert_onnx.cpp)
    engine_test(tinyGPT tinyGPT/tinyGPT.cpp)
    engine_test(slide_tdnn voice_wake_up/slide_tdnn.cpp)
    if (USE_API_CPP)
        include_directories(${BOLT_ROOT}/inference/engine/api/c)
        link_directories(${CMAKE_BINARY_DIR})
        engine_test(cpp_test cpp_api/cpp_test.cpp)
        target_link_libraries(cpp_test bolt_cpp bolt)
        install(TARGETS cpp_test
                RUNTIME DESTINATION examples)
    endif ()
    if (EXISTS ${OpenCV_CMAKE_PATH})
        engine_test(ultra_face ultra_face/ultra_face.cpp)
        engine_test(u2net image_matting/u2net.cpp)
        install(TARGETS ultra_face
                        u2net
                RUNTIME DESTINATION examples)
    endif()
    install(TARGETS classification
                    tinybert
                    tinybert_onnx
                    nmt
                    nmt_en-cs
                    tsc_ssru
                    asr_rnnt
                    asr_convolution_transformer
                    tts
                    vad
                    c_image_classification
                    c_input_method
                    c_set_inout
            RUNTIME DESTINATION examples)

    if (USE_FLOW)
        flow_test(graph_tinybert bert/graph_tinybert.cpp)
        flow_test(flow_tinybert bert/flow_tinybert.cpp)
        flow_test(flow_asr "automatic_speech_recognition/flow_asr.cpp;automatic_speech_recognition/audio_feature.cpp")
        flow_test(flow_dlaWOdcn dlaWOdcn/flow_dlaWOdcn.cpp)
        flow_test(flow_facesr facesr/flow_facesr.cpp)
        install(TARGETS flow_asr
                        flow_dlaWOdcn
                        flow_facesr
                RUNTIME DESTINATION examples)
    endif ()
endif ()
