cmake_minimum_required(VERSION 3.2)
file(GLOB BOLT_CONFIGURE_FILE $ENV{BOLT_ROOT}/bolt.cmake ${BOLT_ROOT}/bolt.cmake)
if (BOLT_CONFIGURE_FILE)
    include(${BOLT_CONFIGURE_FILE})
endif (BOLT_CONFIGURE_FILE)

project(bolt C CXX)

add_subdirectory(blas-enhance)
add_subdirectory(model-tools)
add_subdirectory(tensor_computing)
add_subdirectory(image)
add_subdirectory(engine)

add_dependencies(tensor_computing blas-enhance)
add_dependencies(tensor_computing_static blas-enhance_static)
add_dependencies(model-tools model-tools_caffe)
add_dependencies(model-tools_static model-tools_caffe_static)
add_dependencies(engine tensor_computing model-tools image)
add_dependencies(engine_static tensor_computing_static model-tools_static image_static)

install(TARGETS blas-enhance blas-enhance_static tensor_computing tensor_computing_static
                model-tools model-tools_static model-tools_caffe model-tools_caffe_static
                image image_static
                engine engine_static
                caffe2bolt
                onnx2bolt
                classification
                bert
                tinybert
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)