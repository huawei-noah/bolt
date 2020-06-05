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
#include <limits.h>
#include <float.h>
#include "inference.hpp"
#include "tensor.hpp"
#include "data_loader.hpp"
#include "result_format.hpp"
#include "utils.hpp"

void print_help(char* argv[])
{
    std::cout << "usage: " << argv[0] << " modelPath cpuAffinityPolicyName algorithmMapPath" << std::endl;
}

int main(int argc, char* argv[])
{
    UTIL_TIME_INIT

    char* modelPath = (char*)"";
    char* cpuAffinityPolicyName = (char*)"";
    char* algorithmMapPath = (char*)"";
    DeviceTypeIn device = d_CPU;
    U32 testNum = 1;
    if (argc < 2) {
        print_help(argv);
        return 1;
    }
    modelPath = argv[1];
    if (argc > 2) {
        const char* deviceName = "GPU";
        const char* argvName = argv[2];
        if(strcmp(deviceName, argvName) == 0) {
            device = d_GPU;
        } else {
            cpuAffinityPolicyName = argv[2];
        }
    }
    if (argc > 3) {
        algorithmMapPath = argv[3];
    }
    /* 
    char* imageDir = (char*)"";
    ImageFormat ImageFormat = RGB;
    if (argc > 4) {
        imageDir = argv[2];
    } 
    if (argc > 5) {
        ImageFormat = (std::string(argv[3]) == std::string("BGR") ? BGR : RGB);
        if (std::string(argv[3]) == std::string("RGB_SC")) {
            ImageFormat = RGB_SC;
        } else if (std::string(argv[3]) == std::string("BGR_SC_RAW")) {
            ImageFormat = BGR_SC_RAW;
        } else if (std::string(argv[3]) == std::string("RGB_SC_RAW")) {
            ImageFormat = RGB_SC_RAW;
        }
    }
    */
    auto cnn = createPipelineWithConfigure(cpuAffinityPolicyName, modelPath, device, algorithmMapPath);

    //set input value(TODO use image)
    HashMap<std::string, std::shared_ptr<Tensor>> inMap = cnn->get_inputs();
    TensorDesc imageDesc = (*(inMap.begin()->second)).get_desc();
    Vec<TensorDesc> imageDescs;
    imageDescs.push_back(imageDesc);
    Vec<Vec<Tensor>> images;
    for(U32 i = 0; i < testNum; i++) {
        std::shared_ptr<U8> imageData((U8*) operator new(tensorNumBytes(imageDescs[0])));
        F16* val = (F16*)imageData.get();
        for(U32 i = 0; i < tensorNumElements(imageDescs[0]); i++) val[i] = (i % 1024) / 1024.0 - 0.5;
        std::shared_ptr<Tensor> tensorData(new Tensor());
        tensorData->set_desc(imageDescs[0]);
        tensorData->set_shared_ptr(imageData);
        Vec<Tensor> image;
        image.push_back(*tensorData.get());
        images.push_back(image);
    }
    
    double max_time = -DBL_MAX;
    double min_time = DBL_MAX;
    double totalTime = 0;
    U32 imageIndex = 0;
    U32 invalidIndex = 0;
    std::cout << "[RESULT]:" << std::endl;
    
    for (auto image: images) {
        // stage3: set input
        double timeBegin = ut_time_ms();
        if(device == d_CPU){        
            auto curModelInputTensorNames = cnn->get_model_input_tensor_names();
            for (int index = 0; index < (int)curModelInputTensorNames.size(); index++) {
                cnn->copy_to_named_input(curModelInputTensorNames[index], image[index].get_val());
                DEBUG_info("curModelInputTensorNames[index]: " << curModelInputTensorNames[index]);
            }
        } else {
            auto curModelInputTensorNames = cnn->get_model_input_tensor_names();
            HashMap<std::string, std::shared_ptr<U8>> modelInputTensors;
            for (int index = 0; index < (int)curModelInputTensorNames.size(); index++) {
                modelInputTensors[curModelInputTensorNames[index]] = image[index].get_shared_ptr();
            }
            cnn->set_input_tensors_value(modelInputTensors);
        }
        // stage4: run
        cnn->run();
          
        // stage5: process result
        HashMap<std::string, std::shared_ptr<Tensor>> outMap = cnn->get_outputs();
        double timeEnd = ut_time_ms();
        totalTime += (timeEnd - timeBegin);
        Tensor result = *(outMap.begin()->second);
        bool invalid = result.isInvalid();
        if (!invalid) {
	    if ((timeEnd - timeBegin) >= max_time) {
	        max_time = (timeEnd - timeBegin);
	    }

	    if ((timeEnd - timeBegin) <= min_time) {
	        min_time = (timeEnd - timeBegin);
	    }
        }
        else{
            std::cout << "warnning the result get nan" << std::endl;
            totalTime -=(timeEnd - timeBegin);
            invalidIndex++;
        }
        imageIndex++;
    }

    UTIL_TIME_STATISTICS

    std::cout << "[SUMMARY]:" << std::endl;
    CI_info("avg_time:" << 1.0 * totalTime / (imageIndex - invalidIndex) << "ms/image");
    CI_info("max_time:" << 1.0 * max_time << "ms/image");
    CI_info("min_time:" << 1.0 * min_time << "ms/image");
    return 0;
}
