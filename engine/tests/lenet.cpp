// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <iostream>
#include "type.h"
#include "tensor_desc.h"
#include "sequential.hpp"
#include "factory.hpp"
#include "tensor.hpp"
#include "data_loader.hpp"


void print_help(char* argv[]) {
    std::cout << "usage: " << argv[0] << "  imageDir"<<std::endl;
}


int main(int argc, char* argv[]) {
    char* imageDir = (char*)"";
    if(argc != 2) {
        print_help(argv);
    }
    else
        imageDir = argv[1];

    const Arch A = ARM_A76;
    DataType dt = DT_F16;
    auto model = Sequential<A>(dt, "lenet");

    auto op = Factory::createConvolution<A>(dt, 8, 5, 1, 2, ACTIVATION_NULL, ACTIVATION_NULL, Convolution_Pointwise, 1, 1);
    model.add(op);

    op = Factory::createPooling<A>(PoolingMode::Max, 2, 2, 0, RoundMode::CEIL);
    model.add(op);

    op = Factory::createConvolution<A>(dt, 8, 3, 1, 1, ACTIVATION_NULL, ACTIVATION_NULL, Convolution_Pointwise, 1, 1);
    model.add(op);

    op = Factory::createPooling<A>(PoolingMode::Max, 2, 2, 0, RoundMode::CEIL);
    model.add(op);

    op = Factory::createFullyConnectedEltwise<A>(dt, 10);
    model.add(op);

    op = Factory::createSoftmax<A>(dt);
    model.add(op);

    TensorDesc imageDesc = tensor4df(DT_F16, DF_NCHW, 1, 1, 8, 8);

    auto weight = (F16*)operator new(256*256*256*sizeof(F16));
    for (int i = 0; i < 256*256*256; i++) {
        weight[i] = 1;
    }
    U8* wPtr = (U8*)weight;
    std::shared_ptr<U8> modelPtr(wPtr);
    model.ready({imageDesc}, modelPtr);

    // load images
    Vec<Tensor> images;
    load_images(imageDir, imageDesc, &images, BGR, 1.0);

    for (auto image: images) {
        Vec<Tensor> input;
        input.push_back(image);
        model.set_input_tensors(input);

        model.run();

        auto outputs = model.get_output_tensors();
        outputs[0].print<F16>();
    }

    return 0;
}
