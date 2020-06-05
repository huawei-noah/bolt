cmake_minimum_required(VERSION 3.2)

file(GLOB BOLT_CONFIGURE_FILE $ENV{BOLT_ROOT}/bolt.cmake ${BOLT_ROOT}/bolt.cmake)
if (BOLT_CONFIGURE_FILE)
    include(${BOLT_CONFIGURE_FILE})
else (BOLT_CONFIGURE_FILE)
    message(FATAL_ERROR "
FATAL: can not find bolt.cmake in <BOLT_ROOT> directory,
       please set shell or cmake environment variable BOLT_ROOT.
    ")
endif (BOLT_CONFIGURE_FILE)

project(kits)

set_policy()

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${BOLT_ROOT}/cmakes")
find_package(Uni)
find_package(ModelTools)
find_package(Image)
find_package(TensorComputing)
find_package(Inference)
find_package(jpeg)
if(USE_MALI)
    find_package(Gcl)
endif(USE_MALI)

set_project_install_directory()
set_c_cxx_flags()
set_test_c_cxx_flags()

if (BUILD_TEST)
    inference(bert bert/bert.cpp)
    inference(tinybert bert/tinybert.cpp)
    inference(classification image_classification/classification.cpp)
    inference(classification_bin image_classification/classification_bin.cpp)
    inference(nmt machine_translation/nmt.cpp)
    inference(nmt_tsc machine_translation/nmt_tsc.cpp)
    inference(asr_rnnt automatic_speech_recognition/asr_rnnt.cpp)
    inference(asr_convolution_transformer automatic_speech_recognition/asr_convolution_transformer.cpp)
    inference(tts text_to_speech/tts.cpp)
    inference(vad automatic_speech_recognition/vad.cpp)
    if (USE_MALI)
        if (USE_FP16)
            inference(hdr high_dynamic_range/hdr.cpp)
            inference(super_resolution super_resolution/super_resolution.cpp)
        endif (USE_FP16)
    endif (USE_MALI)
endif (BUILD_TEST)
