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
    std::cout << "usage: " << argv[0] << " modelPath imagesDirectory imageStyle scaleValue topK correctLabel cpuAffinityPolicyName algorithmMapPath" << std::endl;
}

int main(int argc, char* argv[])
{
    UTIL_TIME_INIT

    char* modelPath = (char*)"";
    char* imageDir = (char*)"";
    char* cpuAffinityPolicyName = (char*)"";
    char* algorithmMapPath = (char*)"";
    ImageFormat ImageFormat = RGB;
    DeviceTypeIn device = d_CPU;
    F32 scaleValue = 1;
    int topK = 5;
    int category = -1;
    if (argc < 2) {
        print_help(argv);
        return 1;
    }
    modelPath = argv[1];
    if (argc > 2) {
        imageDir = argv[2];
    } 
    if (argc > 3) {
        ImageFormat = (std::string(argv[3]) == std::string("BGR") ? BGR : RGB);
        if (std::string(argv[3]) == std::string("RGB_SC")) {
            ImageFormat = RGB_SC;
        } else if (std::string(argv[3]) == std::string("BGR_SC_RAW")) {
            ImageFormat = BGR_SC_RAW;
        } else if (std::string(argv[3]) == std::string("RGB_SC_RAW")) {
            ImageFormat = RGB_SC_RAW;
        }
    }
    if (argc > 4) {
        scaleValue = atof(argv[4]);
    }
    if (argc > 5) {
        topK = atoi(argv[5]);
    } 
    if (argc > 6) {
        category = atoi(argv[6]);
    }
    if (argc > 7) {
        const char* deviceName = "GPU";
        const char* argvName = argv[7];
        if(strcmp(deviceName, argvName) == 0) {
            device = d_GPU;
        } else {
            cpuAffinityPolicyName = argv[7];
        }
    }
    if (argc > 8) {
        algorithmMapPath = argv[8];
    }
    //DeviceTypeIn device = d_CPU;
    auto cnn = createPipelineWithConfigure(cpuAffinityPolicyName, modelPath, device, algorithmMapPath);

    // load images
    HashMap<std::string, std::shared_ptr<Tensor>> inMap = cnn->get_inputs();
    TensorDesc imageDesc = (*(inMap.begin()->second)).get_desc();
    Vec<TensorDesc> imageDescs;
    imageDescs.push_back(imageDesc);
    Vec<Vec<Tensor>> images;
    Vec<std::string> imagePaths = load_image_with_scale(imageDir, imageDescs, &images, ImageFormat, scaleValue);

    std::map<int, int> categoryNum;
    double totalTime = 0;
    double max_time = -DBL_MAX;
    double min_time = DBL_MAX;
    U32 imageIndex = 0;
    std::cout << "[RESULT]:" << std::endl;
    int top1Index = 0;
    int top1Match = 0; 
    int topKMatch = 0; 
    const int INVALID_INDEX = INT_MAX;
    
    for (auto image: images) {
        std::cout << imagePaths[imageIndex] << " : ";
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
                std::shared_ptr<U8> tensorPointer = image[index].get_shared_ptr();
                modelInputTensors.insert(std::pair(curModelInputTensorNames[index], tensorPointer));
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
            Vec<int> topKResult = topK_index(result, topK);
            top1Index = topKResult[0];
            if (category != -1) {
                if (top1Index == category) {
                    top1Match ++;
                }
                for (int i = 0; i < topK; i++) {
                    if(topKResult[i] == category) {
                        topKMatch++;
                        break;
                    }
                }
                for (int i = 0; i < topK; i++) {
                    std::cout << topKResult[i] << " ";
                }
                std::cout << std::endl;
            }

	    if ((timeEnd - timeBegin) >= max_time) {
	        max_time = (timeEnd - timeBegin);
	    }

	    if ((timeEnd - timeBegin) <= min_time) {
	        min_time = (timeEnd - timeBegin);
	    }
        }
        else{
            totalTime -= (timeEnd - timeBegin);
            top1Index = INVALID_INDEX;
            std::cout << "nan" << std::endl;
        }
        if (categoryNum.count(top1Index) == 0) {
            categoryNum.insert( std::pair<int, int>(top1Index, 1));
        } else {
            categoryNum[top1Index] = categoryNum[top1Index] + 1;
        }
        imageIndex++;
    }

    UTIL_TIME_STATISTICS

    std::cout << "[CATEGORY]:" << std::endl;
    std::cout << "category\tnum" << std::endl;
    U32 nanImages = 0;
    for (auto elem : categoryNum) {
        if(elem.first == INVALID_INDEX){
            std::cout << "nan\t" << elem.second << std::endl;
            nanImages = elem.second;
        }
        else {
            std::cout << elem.first << "\t" << elem.second << std::endl;
        }
    }
    U32 validImages = imageIndex - nanImages;
    std::cout << "[SUMMARY]:" << std::endl;
    CI_info("top1:" << 1.0 * top1Match / validImages);
    CI_info("top" << topK << ":" << 1.0 * topKMatch / validImages);
    CI_info("avg_time:" << 1.0 * totalTime / validImages << "ms/image");
    CI_info("max_time:" << 1.0 * max_time << "ms/image");
    CI_info("min_time:" << 1.0 * min_time << "ms/image");
    return 0;
}
