cmake_minimum_required(VERSION 3.15)

file(GLOB BOLT_CONFIGURE_FILE $ENV{BOLT_ROOT}/bolt.cmake ${BOLT_ROOT}/bolt.cmake)
if (BOLT_CONFIGURE_FILE)
    include(${BOLT_CONFIGURE_FILE})
else (BOLT_CONFIGURE_FILE)
    message(FATAL_ERROR "
FATAL: can not find bolt.cmake in <BOLT_ROOT> directory,
       please set shell or cmake environment variable BOLT_ROOT.
    ")
endif (BOLT_CONFIGURE_FILE)

project(bolt C CXX)

if (USE_MALI)
    add_subdirectory(gcl/tools/kernel_lib_compile)
endif (USE_MALI)
add_subdirectory(blas-enhance)
add_subdirectory(model-tools)
add_subdirectory(tensor_computing)
add_subdirectory(image)
add_subdirectory(inference)
add_subdirectory(tests)
add_custom_target(bolt_library ALL
    COMMAND ./scripts/build_light_bolt.sh ${CMAKE_BINARY_DIR} ${USE_MALI}
    WORKING_DIRECTORY $ENV{BOLT_ROOT})

if (USE_MALI)
    add_dependencies(inference kernelbin)
    add_dependencies(inference_static kernelbin_static)
    install(TARGETS kernelbin kernelbin_static
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib)
    install(FILES ${CMAKE_BINARY_DIR}/libOpenCL.so
            DESTINATION lib)
endif (USE_MALI)
add_dependencies(tensor_computing blas-enhance)
add_dependencies(tensor_computing_static blas-enhance_static)
add_dependencies(inference tensor_computing model-tools image)
add_dependencies(inference_static tensor_computing_static model-tools_static image_static)
add_dependencies(bolt_library inference_static)

install(TARGETS blas-enhance blas-enhance_static
                tensor_computing tensor_computing_static
                model-tools model-tools_static
                image image_static
                inference inference_static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

if (USE_CAFFE)
    add_dependencies(model-tools model-tools_caffe)
    add_dependencies(model-tools_static model-tools_caffe_static)
    install(TARGETS caffe2bolt
                    model-tools_caffe model-tools_caffe_static
            RUNTIME DESTINATION tools
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib)
endif(USE_CAFFE)

if (USE_ONNX)
    add_dependencies(model-tools model-tools_onnx)
    add_dependencies(model-tools_static model-tools_onnx_static)
    install(TARGETS onnx2bolt
                    model-tools_onnx model-tools_onnx_static
            RUNTIME DESTINATION tools
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib)
endif(USE_ONNX)

if (USE_TFLITE)
    add_dependencies(model-tools model-tools_tflite)
    add_dependencies(model-tools_static model-tools_tflite_static)
    install(TARGETS tflite2bolt
                    model-tools_tflite model-tools_tflite_static
            RUNTIME DESTINATION tools
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib)
endif(USE_TFLITE)

install(DIRECTORY model-tools/tools/tensorflow2caffe
                  model-tools/tools/pytorch2caffe
        DESTINATION tools)

install(TARGETS tensor_computing_library_search
        RUNTIME DESTINATION tools)

if (BUILD_TEST)
    install(TARGETS classification
                    bert
                    tinybert
                    nmt
            RUNTIME DESTINATION bin)
endif(BUILD_TEST)

install(DIRECTORY inference/exports/java
                  inference/exports/c
        DESTINATION include)

install(FILES ${CMAKE_BINARY_DIR}/libBoltModel.so
        DESTINATION lib)

execute_process(COMMAND doxygen .Doxyfile WORKING_DIRECTORY $ENV{BOLT_ROOT})

enable_testing()

find_program (BASH_PROGRAM bash)

if (BASH_PROGRAM)
    set(parameters -b $ENV{BOLT_ROOT}/tests/bin -p /data/local/tmp/uldra)
    if (USE_MALI)
        set(parameters ${parameters} -g)
    endif(USE_MALI)
  
    if (USE_DYNAMIC_LIBRARY)
        set(parameters ${parameters} -l $ENV{BOLT_ROOT}/install_llvm/lib)
    endif(USE_DYNAMIC_LIBRARY)
    add_test (NAME quick_benchmark COMMAND $ENV{BOLT_ROOT}/quick_benchmark.sh ${parameters})
endif (BASH_PROGRAM)
