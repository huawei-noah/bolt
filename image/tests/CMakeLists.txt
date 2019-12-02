function(b_1_b name)
    add_executable(${name} ${name}.cpp)
    add_dependencies(${name} ${PROJECT_NAME}_static)
    target_link_libraries(${name} ${IMAGE_LIBRARIES})
endfunction()

if (USE_CROSS_COMPILE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static")
endif (USE_CROSS_COMPILE)

b_1_b(test_image_processing)
