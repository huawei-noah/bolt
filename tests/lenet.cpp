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
#include "inference.hpp"
#include "factory.hpp"
#include "cpu/factory_cpu.hpp"
#include "tensor.hpp"
#include "data_loader.hpp"


void print_help(char* argv[]) {
    std::cout << "usage: " << argv[0] << " inferencePrecision cpuAffinityPolicyName imageDirectory" << std::endl;
}

template<typename T>
int lenet(int argc, char* argv[], DataType dt)
{
    char* imageDir = (char*)"";
    char* cpuAffinityPolicyName = (char*)"";
 
    if (argc > 2) {
        cpuAffinityPolicyName = argv[2];
    }
    if (argc > 3) {
        imageDir = argv[3];
    }

    auto model = createSequentialPipeline(cpuAffinityPolicyName, dt, "lenet");

    Factory* factory_cpu = (Factory*)(new FactoryCPU());
    std::shared_ptr<Factory> factory;
    factory = std::shared_ptr<Factory>(factory_cpu);

    auto op = factory->createConvolution(dt, 8, 5, 5, 1, 1, 2, 2, 2, 2, ACTIVATION_NULL, ACTIVATION_NULL, Convolution_Pointwise, 1, 1, 1);
    model.add(op);

    op = factory->createPooling(PoolingMode::POOLING_MAX, 2, 2, 2, 2, 0, 0, 0, 0, RoundMode::CEIL);
    model.add(op);

    op = factory->createConvolution(dt, 8, 3, 3, 1, 1, 1, 1, 1, 1, ACTIVATION_NULL, ACTIVATION_NULL, Convolution_Pointwise, 1, 1, 1);
    model.add(op);

    op = factory->createPooling(PoolingMode::POOLING_MAX, 2, 2, 2, 2, 0, 0, 0, 0, RoundMode::CEIL);
    model.add(op);

    op = factory->createFullyConnectedEltwise(dt, 32, 10);
    model.add(op);

    op = factory->createSoftmax(dt);
    model.add(op);

    TensorDesc imageDesc = tensor4df(dt, DF_NCHW, 1, 1, 8, 8);

    auto weight = (T*)operator new(256*256*256*sizeof(T));
    for (int i = 0; i < 256*256*256; i++) {
        weight[i] = 1;
    }
    U8* wPtr = (U8*)weight;
    std::shared_ptr<U8> modelPtr(wPtr);
    model.initialize_weight(modelPtr);
    model.ready({imageDesc});

     // load images
    Vec<TensorDesc> imageDescs;
    imageDescs.push_back(imageDesc);
    Vec<Vec<Tensor>> images;
    Vec<std::string> imagePaths = load_data(imageDir, imageDescs, &images);

    double totalTime = 0;
    for (auto image: images) {
        model.set_input_tensors(image);

        double timeBegin = ut_time_ms();
        model.run();
        double timeEnd = ut_time_ms();
        totalTime += (timeEnd - timeBegin);

        auto outputs = model.get_output_tensors();
        std::cout << "Network outputs (for quick_benchmark they should be the same value)\n";
        outputs[0].print();
    }

    UTIL_TIME_STATISTICS
    std::cout << "=== Total time is " << totalTime << " ms for " << images.size() << " image(s)\n";

    return 0;
}

int main(int argc, char* argv[]) {
    if (argc == 1) {
        print_help(argv);
    }
    if (std::string(argv[1]) == std::string("FP16")) {
#ifdef _USE_FP16
        std::cout << "\n=== FP16:\n";
        return lenet<F16>(argc, argv, DT_F16);
#endif
    } else if (std::string(argv[1]) == std::string("FP32")) {
#ifdef _USE_FP32
        std::cout << "\n=== FP32:\n";
        return lenet<F32>(argc, argv, DT_F32);
#endif
    }
    return 1;
}
