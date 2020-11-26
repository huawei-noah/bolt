function(blas_enhance_test name)
    add_executable(${name} ${name}.cpp)
    link_blas_enhance(${name})
    install(TARGETS ${name} 
        RUNTIME DESTINATION tests)
endfunction()

set_test_c_cxx_flags()

blas_enhance_test(test_mmm)
blas_enhance_test(test_mvm)
blas_enhance_test(test_mmm_int8)
blas_enhance_test(test_mvm_int8)
