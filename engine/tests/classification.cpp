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
#include "inference.hpp"
#include "tensor.hpp"
#include "type.h"
#include "tensor_desc.h"
#include "model_serialize_deserialize.hpp"
#include "data_loader.hpp"
#include "result_format.hpp"
#include "utils.hpp"

void print_help(char* argv[])
{
    std::cout << "usage: " << argv[0] << "  model_path  images_dir  image_style  scale_value  topK  correct_label" << std::endl;
}

int main(int argc, char* argv[])
{
    UTIL_TIME_INIT

    char* modelPath = (char*)"";
    char* imageDir = (char*)"";
    ImageType imageType = RGB;
    F32 scaleValue = 1;
    int topK = 5;
    int category = -1;
    const Arch A = ARM_A76;
    
    if (argc < 2) {
        print_help(argv);
        return 1;
    }
    modelPath = argv[1];
    if (argc > 2) {
        imageDir = argv[2];
    } 
    if (argc > 3) {
        imageType = (std::string(argv[3]) == std::string("BGR") ? BGR : RGB);
        if (std::string(argv[3]) == std::string("RGB_SC")) {
            imageType = RGB_SC;
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
    
    ModelSpec ms;
    deserialize_model_from_file(modelPath, &ms);
    auto cnn = createCNN<A>(&ms);

    // load images
    Vec<Tensor> images;
    HashMap<std::string, std::shared_ptr<Tensor>> inMap = cnn->get_inputs();
    TensorDesc imageDesc = (*(inMap.begin()->second)).get_desc();
    Vec<std::string> imagePaths = load_images(imageDir, imageDesc, &images, imageType, scaleValue);

    std::map<int, int> categoryNum;
    double totalTime = 0;
    U32 imageIndex = 0;
    std::cout << "[RESULT]:" << std::endl;
    int top1Index = 0;
    int top1Match = 0; 
    int topKMatch = 0; 
    const int INVALID_INDEX = INT_MAX;
    for (auto image: images) {
        std::cout << imagePaths[imageIndex] << " : ";
        // stage3: set input
        Vec<Tensor> input;
        input.push_back(image);
        auto curModelInputTensorNames = cnn->get_model_input_tensor_names();
        HashMap<std::string, std::shared_ptr<U8>> modelTensorsInput;
        for (int index = 0; index < (int)curModelInputTensorNames.size(); index++) {
            std::shared_ptr<U8> tensorPtrTemp = input[index].get_val();
            modelTensorsInput.insert(std::pair(curModelInputTensorNames[index], tensorPtrTemp));
#ifdef _DEBUG
            std::cerr << "curModelInputTensorNames[index]: " << curModelInputTensorNames[index] << std::endl;
            std::cerr << "tensorPtrTemp.get(): " << (void*)tensorPtrTemp.get() << std::endl;
#endif
        }
        cnn->set_input_tensors_value(modelTensorsInput);

        // stage4: run
        double timeBegin = ut_time_ms();
        cnn->run();
        double timeEnd = ut_time_ms();
        totalTime += (timeEnd - timeBegin);

        // stage5: process result
        HashMap<std::string, std::shared_ptr<Tensor>> outMap = cnn->get_outputs();
        Tensor result = *(outMap.begin()->second);
        bool invalid = result.isInvalid<F16>();
        if (!invalid) {
            Vec<int> topKResult = topK_index<F16>(result.get_val().get(), tensorNumElements(result.get_desc()), topK);
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
    std::cout << "topK(1): " << top1Match << " matches, which is " << 100.0 * top1Match / validImages << "% " <<std::endl;
    std::cout << "topK(" << topK << "): " << topKMatch << " matches, which is " << 100.0 * topKMatch / validImages << "% " << std::endl;
    std::cout << "time: " << 1.0 * totalTime / validImages << " ms/image" << std::endl;

    return 0;
}
