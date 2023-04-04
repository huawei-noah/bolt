function(image_test name)
    add_executable(${name} ${name}.cpp)
    link_image(${name})
    install(TARGETS ${name}
        RUNTIME DESTINATION tests)
endfunction()

set_test_c_cxx_flags()

if (BUILD_TEST)
    image_test(test_image_processing)
    image_test(test_image_resize)
    if (USE_GPU)
        image_test(test_image_resize_ocl test_image_resize_ocl.cpp)
    endif ()
endif ()
