if (USE_GENERAL)
    file(GLOB general_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/general/*.cpp)
endif (USE_GENERAL)

if (USE_X86)
    file(GLOB x86_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/x86/*.cpp)
    if (USE_FP32)
        file(GLOB x86_fp32_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/x86/fp32/*.cpp)
    endif (USE_FP32)
    set(x86_srcs "${x86_srcs};${x86_fp32_srcs};")
endif (USE_X86)

if (USE_NEON)
    if (USE_FP16)
        file(GLOB arm_fp16_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/arm/fp16/*.cpp)
    endif (USE_FP16)
    if (USE_FP32)
        file(GLOB arm_fp32_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/arm/fp32/*.cpp)
        if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
            file(GLOB armv8_fp32_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/arm/fp32/v8/*.cpp)
        else ()
            file(GLOB armv7_fp32_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/arm/fp32/v7/*.cpp)
        endif ()
        set(arm_fp32_srcs "${arm_fp32_srcs};${armv8_fp32_srcs};${armv7_fp32_srcs}")
    endif (USE_FP32)
    if (USE_INT8)
        file(GLOB arm_int8_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/arm/int8/*.cpp)
    endif (USE_INT8)
    file(GLOB arm_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/arm/*.cpp)
    set(arm_srcs "${arm_srcs};${arm_fp16_srcs};${arm_fp32_srcs};${arm_int8_srcs}")
endif (USE_NEON)

file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(srcs "${srcs};${general_srcs};${arm_srcs};${x86_srcs}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# shared library 
add_library(${PROJECT_NAME} SHARED ${srcs})
target_link_libraries(${PROJECT_NAME} LINK_PUBLIC uni)

# static library 
add_library(${PROJECT_NAME}_static STATIC ${srcs})

set_target_properties(${PROJECT_NAME}_static PROPERTIES OUTPUT_NAME "${PROJECT_NAME}")
set_target_properties(${PROJECT_NAME} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(${PROJECT_NAME}_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
