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

if (USE_IOS_CLANG)
    set(CMAKE_SYSTEM_NAME Darwin)
    set(CMAKE_SYSTEM_VERSION 1)
    set(UNIX True)
    set(APPLE True)
    set(IOS True)
endif (USE_IOS_CLANG)

project(cheetah C CXX)

set_policy()
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${BOLT_ROOT}/common/cmakes")
if (USE_CAFFE OR USE_ONNX OR USE_FLOW)
    find_package(Protobuf)
endif()
if (USE_TFLITE)
    find_package(TFLite)
endif (USE_TFLITE)
if (USE_TENSORFLOW)
    find_package(jsoncpp)
endif (USE_TENSORFLOW)
if (USE_MALI)
    find_package(Gcl)
endif(USE_MALI)
if (NOT USE_IOS_CLANG)
    if (USE_LLVM_CLANG)
        set(USE_JNI ON)
    else()
        find_package(JNI)
        if (JNI_FOUND)
            set(USE_JNI ON)
        endif()
    endif()
endif()

add_subdirectory(common)
add_subdirectory(model_tools)
add_subdirectory(compute)
add_subdirectory(inference)
add_custom_target(bolt_library ALL
    COMMAND ./scripts/build_light_bolt.sh ${CMAKE_CXX_COMPILER} ${CMAKE_AR} ${CMAKE_STRIP} ${CMAKE_BINARY_DIR} ${USE_MALI} ${USE_DEBUG} ${USE_LLVM_CLANG} ${USE_ANDROID_LOG} ${USE_IOS_CLANG} ${USE_OPENMP}
    WORKING_DIRECTORY $ENV{BOLT_ROOT})
add_dependencies(bolt_library engine model_tools tensor image blas_enhance uni)
add_dependencies(bolt_library engine_static model_tools_static tensor_static image_static blas_enhance_static uni_static)

# install section
install(FILES ${CMAKE_BINARY_DIR}/libbolt.a
        DESTINATION lib)
if (USE_IOS_CLANG)
    install(FILES ${CMAKE_BINARY_DIR}/libbolt.dylib
        DESTINATION lib)
else (USE_IOS_CLANG)
    install(FILES ${CMAKE_BINARY_DIR}/libbolt.so
        DESTINATION lib)
    install(FILES ${CMAKE_BINARY_DIR}/libBoltModel.so
        DESTINATION lib)
endif (USE_IOS_CLANG)

execute_process(COMMAND doxygen .Doxyfile WORKING_DIRECTORY $ENV{BOLT_ROOT})

enable_testing()
find_program (BASH_PROGRAM bash)
if (BASH_PROGRAM AND USE_GENERAL)
    set(parameters -t ${CMAKE_INSTALL_PREFIX} -p /data/local/tmp/uldra)
    if (USE_MALI)
        set(parameters ${parameters} -g)
    endif(USE_MALI)
    if (USE_DYNAMIC_LIBRARY)
        set(parameters ${parameters} -l ${CMAKE_INSTALL_PREFIX}/lib)
    endif(USE_DYNAMIC_LIBRARY)
    add_test (NAME quick_benchmark COMMAND $ENV{BOLT_ROOT}/scripts/quick_benchmark.sh ${parameters})
endif (BASH_PROGRAM AND USE_GENERAL)
