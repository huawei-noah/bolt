if (USE_LITE)
    set(srcs "convolution.cpp;"
        "eltwise.cpp;scale.cpp;power.cpp;"
        "depthwise_convolution.cpp;depthwise_pointwise_convolution.cpp;"
        "activation.cpp;"
        "pooling.cpp;"
        "fully_connected.cpp;"
        "reshape.cpp;"
        #"concat.cpp;"
        )
    if (USE_INT8)
        set(srcs "${srcs};quantize.cpp;dequantize.cpp")
    endif ()
else ()
    file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
endif ()

if (USE_GENERAL OR USE_NEON OR USE_X86)
    if (USE_LITE)
        set(cpu_srcs "cpu/convolution.cpp;"
            "cpu/eltwise.cpp;cpu/scale.cpp;cpu/power.cpp;"
            "cpu/depthwise_convolution.cpp;cpu/depthwise_pointwise_convolution.cpp;"
            "cpu/activation.cpp;"
            "cpu/reshape.cpp;"
            #"cpu/concat.cpp;"
            )
        if (USE_INT8)
            set(cpu_srcs "${cpu_srcs};cpu/quantize.cpp")
        endif ()
    else ()
        file(GLOB cpu_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/*.cpp)
    endif ()
endif ()

if (USE_GENERAL)
    if (USE_LITE)
        set(general_srcs "cpu/general/eltwise.cpp;cpu/general/scale.cpp;"
            "cpu/general/depthwise_convolution.cpp;cpu/general/depthwise_pointwise_convolution.cpp;"
            "cpu/general/pooling.cpp"
            )
        if (USE_INT8)
            set(general_srcs "${general_srcs};cpu/general/dequantize.cpp")
            set(general_srcs "${general_srcs};cpu/general/convolution.cpp")
        else ()
            set(general_srcs "${general_srcs};cpu/lite/convolution.cpp")
        endif ()
    else ()
        file(GLOB general_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/general/*.cpp)
    endif ()
endif ()

if (USE_NEON)
    if (USE_FP32)
        if (USE_LITE)
            set(arm_fp32_srcs "cpu/arm/fp32/convolution.cpp;"
                "cpu/arm/fp32/convolution_transform.cpp;"
                "cpu/arm/fp32/depthwise_pointwise_convolution.cpp;"
                "cpu/arm/fp32/eltwise.cpp;"
                "cpu/arm/fp32/pooling.cpp;"
                "cpu/arm/fp32/scale.cpp"
                )
            if (USE_INT8)
                set(arm_fp32_srcs "${arm_fp32_srcs};cpu/arm/fp32/dequantize.cpp")
            endif ()
        else ()
            file(GLOB arm_fp32_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/arm/fp32/*.cpp)
        endif ()
        if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
            file(GLOB armv8_fp32_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/arm/fp32/v8/*.cpp)
        else ()
            file(GLOB armv7_fp32_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/arm/fp32/v7/*.cpp)
        endif ()
        set(arm_fp32_srcs "${arm_fp32_srcs};${armv8_fp32_srcs};${armv7_fp32_srcs}")
    endif (USE_FP32)
    if (USE_FP16)
        file(GLOB arm_fp16_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/arm/fp16/*.cpp)
        file(GLOB arm_bnn_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/arm/bnn/*.cpp)
    endif (USE_FP16)
    if (USE_INT8)
        file(GLOB arm_int8_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/arm/int8/*.cpp)
        if (USE_FP16)
            file(GLOB armv82_int8_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/arm/int8/v8.2/*.cpp)
        elseif ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
            file(GLOB armv8_int8_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/arm/int8/v8/*.cpp)
        else ()
            file(GLOB armv7_int8_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/arm/int8/v7/*.cpp)
        endif ()
        set(arm_int8_srcs "${arm_int8_srcs};${armv9_int8_srcs};${armv82_int8_srcs};${armv8_int8_srcs};${armv7_int8_srcs}")
    endif (USE_INT8)
    if (USE_LITE)
        set(arm_srcs "cpu/arm/convolution.cpp;"
            "cpu/arm/depthwise_convolution.cpp;cpu/arm/depthwise_pointwise_convolution.cpp;"
            "cpu/arm/eltwise.cpp;"
            "cpu/arm/pooling.cpp;"
            "cpu/arm/scale.cpp"
            )
        if (USE_INT8)
            set(arm_srcs "${arm_srcs};cpu/arm/dequantize.cpp")
        endif ()
    else ()
        file(GLOB arm_int32_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/arm/int32/*.cpp)
        file(GLOB arm_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/arm/*.cpp)
    endif ()
    set(arm_srcs "${arm_srcs};${arm_fp16_srcs};${arm_fp32_srcs};${arm_int8_srcs};${arm_bnn_srcs};${arm_int32_srcs}")
endif (USE_NEON)

if (USE_X86)
    file(GLOB x86_int32_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/x86/int32/*.cpp)
    if (USE_FP32)
        file(GLOB x86_fp32_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/x86/fp32/*.cpp)
    endif (USE_FP32)
    if (USE_INT8)
        file(GLOB x86_int8_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/x86/int8/*.cpp)
        if (USE_AVX512_VNNI)
            file(GLOB avx512_int8_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/x86/int8/avx512/*.cpp)
        else ()
            file(GLOB avx_int8_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/x86/int8/avx/*.cpp)
        endif ()
        set(x86_int8_srcs "${x86_int8_srcs};${avx512_int8_srcs};${avx_int8_srcs};")
    endif (USE_INT8)
    file(GLOB x86_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/x86/*.cpp)
    set(x86_srcs "${x86_srcs};${x86_int32_srcs};${x86_fp32_srcs};${x86_int8_srcs}")
endif (USE_X86)

if (USE_GPU)
    file(GLOB mali_fp16_srcs ${CMAKE_CURRENT_SOURCE_DIR}/gpu/mali/fp16/*.cpp)
    file(GLOB mali_uchar_srcs ${CMAKE_CURRENT_SOURCE_DIR}/gpu/mali/uchar/*.cpp)
    file(GLOB mali_srcs ${CMAKE_CURRENT_SOURCE_DIR}/gpu/mali/*.cpp)
    set(mali_srcs "${mali_srcs};${mali_fp16_srcs};${mali_uchar_srcs}")
endif (USE_GPU)

set(srcs "${srcs};${general_srcs};${arm_srcs};${cpu_srcs};${mali_srcs};${x86_srcs}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# shared library 
add_library(${PROJECT_NAME} SHARED ${srcs})
target_link_libraries(${PROJECT_NAME} blas_enhance)
link_uni_shared(${PROJECT_NAME})
link_gcl(${PROJECT_NAME})

# static library 
add_library(${PROJECT_NAME}_static STATIC ${srcs})

set_target_properties(${PROJECT_NAME}_static PROPERTIES OUTPUT_NAME "${PROJECT_NAME}")
set_target_properties(${PROJECT_NAME} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(${PROJECT_NAME}_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION lib
        LIBRARY DESTINATION lib)
