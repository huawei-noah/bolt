function(b_1_b name)
    add_executable(${name} ${name}.cpp)
    add_dependencies(${name} ${PROJECT_NAME}_static)
    target_link_libraries(${name} ${BLAS_ENHANCE_LIBRARY})
endfunction()

if (USE_CROSS_COMPILE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static")
endif(USE_CROSS_COMPILE)


if (USE_NEON)
    b_1_b(test_mmm)
    b_1_b(test_mvm)
    if (USE_INT8)
        b_1_b(test_mmm_int8)
        b_1_b(test_mvm_int8)
    endif (USE_INT8)
endif (USE_NEON)
