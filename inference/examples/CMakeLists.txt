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

engine_test(benchmark benchmark/benchmark.cpp)
engine_test(test_api_c c_api/test_api_c.c)
install(TARGETS benchmark
                test_api_c
        RUNTIME DESTINATION examples)
if (BUILD_TEST)
    engine_test(bert bert/bert.cpp)
    engine_test(tinybert bert/tinybert.cpp)
    engine_test(classification image_classification/classification.cpp)
    engine_test(nmt machine_translation/nmt.cpp)
    engine_test(nmt_tsc machine_translation/nmt_tsc.cpp)
    engine_test(asr_rnnt automatic_speech_recognition/asr_rnnt.cpp)
    engine_test(asr_convolution_transformer automatic_speech_recognition/asr_convolution_transformer.cpp)
    engine_test(tts text_to_speech/tts.cpp)
    engine_test(vad automatic_speech_recognition/vad.cpp)
    engine_test(detection object_detection/detection.cpp)
    engine_test(tinybert_onnx bert/tinybert_onnx.cpp)
    engine_test(tinyGPT tinyGPT/tinyGPT.cpp)
    engine_test(slide_tdnn voice_wake_up/slide_tdnn.cpp)
    install(TARGETS classification
                    tinybert
                    tinybert_onnx
                    nmt
                    asr_rnnt
                    asr_convolution_transformer
                    tts
                    vad
		    tinyGPT
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
    endif (USE_FLOW)
endif (BUILD_TEST)
