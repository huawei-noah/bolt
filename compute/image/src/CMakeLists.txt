if (USE_GENERAL)
    file(GLOB general_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/general/*.cpp)
    file(GLOB cpu_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/*.cpp)
endif (USE_GENERAL)

if (USE_NEON)
    file(GLOB arm_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/arm/*.cpp)
    file(GLOB cpu_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/*.cpp)
endif (USE_NEON)

if (USE_X86)
    file(GLOB x86_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/x86/*.cpp)
    file(GLOB cpu_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/*.cpp)
endif (USE_X86)

if (USE_GPU)
    file(GLOB mali_srcs ${CMAKE_CURRENT_SOURCE_DIR}/gpu/mali/*.cpp)
    file(GLOB mali_fp16_srcs ${CMAKE_CURRENT_SOURCE_DIR}/gpu/mali/fp16/*.cpp)
endif (USE_GPU)

file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(srcs "${srcs};${cpu_srcs};${general_srcs};${arm_srcs};${x86_srcs};${mali_srcs};${mali_fp16_srcs}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# shared library 
add_library(${PROJECT_NAME} SHARED ${srcs})
target_link_libraries (${PROJECT_NAME} LINK_PUBLIC uni)
if (USE_SECURE_C)
    target_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${SecureC_SHARED_LIBRARY})
endif ()

# static library 
add_library(${PROJECT_NAME}_static STATIC ${srcs})

set_target_properties(${PROJECT_NAME}_static PROPERTIES OUTPUT_NAME "${PROJECT_NAME}")
set_target_properties(${PROJECT_NAME} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(${PROJECT_NAME}_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
