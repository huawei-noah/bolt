set(srcs "model.cpp;cnn.cpp;bolt_c.cpp;bolt_c_simplify.cpp")
if (NOT USE_LITE)
    set (srcs "${srcs};tdnn_fully_connected_cpu.cpp")
endif ()
add_library(${PROJECT_NAME} SHARED ${srcs})
add_library(${PROJECT_NAME}_static STATIC ${srcs})
link_directories(${CMAKE_BINARY_DIR})
target_link_libraries(${PROJECT_NAME} tensor image model_spec)
link_uni_shared(${PROJECT_NAME})
link_gcl(${PROJECT_NAME})
if (BUILD_EXAMPLE)
    target_link_libraries(${PROJECT_NAME} ${JPEG_SHARED_LIBRARY})
endif ()
set_target_properties(${PROJECT_NAME}_static PROPERTIES OUTPUT_NAME "${PROJECT_NAME}")
set_target_properties(${PROJECT_NAME} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(${PROJECT_NAME}_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION lib
        LIBRARY DESTINATION lib)

message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
add_custom_target(bolt_library ALL
    COMMAND bash ./scripts/build_light_bolt.sh ${CMAKE_SYSTEM_NAME} ${CMAKE_CXX_COMPILER} ${CMAKE_AR} ${CMAKE_STRIP} ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_OUTPUT_EXTENSION} ${CMAKE_SHARED_LIBRARY_PREFIX} ${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_STATIC_LIBRARY_PREFIX} ${CMAKE_STATIC_LIBRARY_SUFFIX} ${CMAKE_BINARY_DIR} ${USE_DYNAMIC_LIBRARY}
    WORKING_DIRECTORY ${BOLT_ROOT})
add_dependencies(bolt_library engine model_spec tensor image blas_enhance uni)
add_dependencies(bolt_library engine_static model_spec_static tensor_static image_static blas_enhance_static uni_static)
install(FILES ${CMAKE_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}bolt${CMAKE_SHARED_LIBRARY_SUFFIX}
    ${CMAKE_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}bolt${CMAKE_STATIC_LIBRARY_SUFFIX}
    DESTINATION lib)
if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    install(FILES ${CMAKE_BINARY_DIR}/BoltModel.lib ${CMAKE_BINARY_DIR}/bolt.lib
        DESTINATION lib)
endif ()

if (USE_API_JAVA)
    set(lib "bolt_jni")
    set(srcs "bolt_jni.cpp")
    add_library(${lib} SHARED ${srcs})
    add_library(${lib}_static STATIC ${srcs})
    link_directories(${CMAKE_BINARY_DIR})
    target_link_libraries(${lib} bolt)
    link_gcl(${lib})
    link_system_static(${lib})
    set_target_properties(${lib}_static PROPERTIES OUTPUT_NAME "${lib}")
    set_target_properties(${lib} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
    set_target_properties(${lib}_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
    add_dependencies(${lib} bolt_library)
    install(TARGETS ${lib} ${lib}_static
            RUNTIME DESTINATION lib
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib)
endif()

if (USE_API_CPP)
    set(lib "bolt_cpp")
    set(srcs "bolt_cpp.cpp")
    add_library(${lib} SHARED ${srcs})
    add_library(${lib}_static STATIC ${srcs})
    include_directories(${BOLT_ROOT}/inference/engine/api/c)
    link_directories(${CMAKE_BINARY_DIR})
    target_link_libraries(${lib} bolt)
    link_model_tools_shared(${lib})
    link_system_static(${lib})
    set_target_properties(${lib}_static PROPERTIES OUTPUT_NAME "${lib}")
    set_target_properties(${lib} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
    set_target_properties(${lib}_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
    add_dependencies(${lib} bolt_library)
    install(TARGETS ${lib} ${lib}_static
            RUNTIME DESTINATION lib
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib)
endif()

if (USE_API_PYTHON)
    set(lib "bolt_python")
    set(srcs "bolt_python.cpp")
    find_package(pybind11 CONFIG REQUIRED)
    include_directories(${pybind11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
    link_directories(${CMAKE_BINARY_DIR})
    pybind11_add_module(${lib} ${srcs})
    target_link_libraries(${lib} PRIVATE bolt_cpp bolt)
    set_target_properties(${lib} PROPERTIES OUTPUT_NAME "bolt")
    set_target_properties(${lib} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
    add_dependencies(${lib} bolt_library bolt_cpp)
    install(TARGETS ${lib}
            RUNTIME DESTINATION lib
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib)
    if (USE_CAFFE OR USE_ONNX OR USE_FLOW)
        file(GLOB_RECURSE srcs ${Protobuf_ROOT}/*.so*)
        install(FILES ${srcs}
            DESTINATION lib)
    endif()
    if (USE_TENSORFLOW)
        file(GLOB_RECURSE srcs ${JSONCPP_ROOT}/*.so*)
        install(FILES ${srcs}
            DESTINATION lib)
    endif (USE_TENSORFLOW)
endif()

