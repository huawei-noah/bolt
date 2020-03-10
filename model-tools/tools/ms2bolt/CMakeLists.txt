project(model-tools C CXX)

function(b_1_b name)
    add_executable(${name} ${name}.cpp)
    TARGET_LINK_LIBRARIES(${name} ${PROJECT_NAME}_static) 
endfunction()

b_1_b(caffe2mt)
b_1_b(mt2caffe)

b_1_b(tensorflow2mt)
b_1_b(mt2tensorflow)
