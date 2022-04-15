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
#include "profiling.h"
#include "parse_command.h"

int main(int argc, char *argv[])
{
    UNI_TIME_INIT
    ParseRes parse_res;
    parseCommandLine(argc, argv, &parse_res, "examples");

    char *modelPath = (char *)"";
    char *imageDir = (char *)"";
    char *affinityPolicyName = (char *)"CPU_AFFINITY_HIGH_PERFORMANCE";
    char *algorithmMapPath = (char *)"";
    ImageFormat imageFormat = RGB;
    F32 scaleValue = 1;
    if (!parse_res.model.second) {
        exit(-1);
    }
    if (parse_res.model.second) {
        modelPath = parse_res.model.first;
    }
    if (parse_res.inputPath.second) {
        imageDir = parse_res.inputPath.first;
    }
    if (parse_res.archInfo.second) {
        affinityPolicyName = parse_res.archInfo.first;
    }
    if (parse_res.algoPath.second) {
        algorithmMapPath = parse_res.algoPath.first;
    }
    if (parse_res.imageFormat.second) {
        imageFormat = parse_res.imageFormat.first;
    }
    if (parse_res.scaleValue.second) {
        scaleValue = parse_res.scaleValue.first;
    }

    F32 confidenceThreshold = 0.4;

    auto cnn = createPipeline(affinityPolicyName, modelPath, algorithmMapPath);

    // load images
    std::map<std::string, std::shared_ptr<Tensor>> inMap = cnn->get_input();
    TensorDesc imageDesc = (*(inMap.begin()->second)).get_desc();
    std::vector<TensorDesc> imageDescs;
    imageDescs.push_back(imageDesc);
    std::vector<std::vector<Tensor>> images;
    std::vector<std::string> imagePaths =
        load_image_with_scale(imageDir, imageDescs, &images, imageFormat, scaleValue);

    double totalTime = 0;
    double max_time = -DBL_MAX;
    double min_time = DBL_MAX;
    U32 imageIndex = 0;
    std::cout << "[RESULT]:" << std::endl;

    for (auto image : images) {
        std::cout << imagePaths[imageIndex] << " : ";
        // stage3: set input
        double timeBegin = ut_time_ms();
        std::map<std::string, U8 *> inputMap;
        for (auto iter : inMap) {
            inputMap[iter.first] = (U8 *)((CpuMemory *)(image[0].get_memory()))->get_ptr();
        }
        cnn->set_input_by_copy(inputMap);

        // stage4: run
        cnn->run();

        // stage5: process result
        std::map<std::string, std::shared_ptr<Tensor>> outMap = cnn->get_output();
        double timeEnd = ut_time_ms();
        totalTime += (timeEnd - timeBegin);
        Tensor result = *(outMap.begin()->second);
        F32 numBox = result.element(0);
        std::cout << numBox << " boxes in total, including these ones with confidence over "
                  << confidenceThreshold << ":\n";

        for (U32 i = 6; i < result.length(); i++) {
            F32 confidence = result.element(i + 1);
            if (confidence < confidenceThreshold) {
                break;
            }
            F32 label = result.element(i);
            F32 topLeftX = result.element(i + 2);
            F32 topLeftY = result.element(i + 3);
            F32 bottomRightX = result.element(i + 4);
            F32 bottomRightY = result.element(i + 5);
            std::cout << "\tClass " << label << " with " << confidence << " confidence, top left ("
                      << topLeftX << ", " << topLeftY << ")"
                      << ", bottom right (" << bottomRightX << ", " << bottomRightY << ")\n";
        }

        if ((timeEnd - timeBegin) >= max_time) {
            max_time = (timeEnd - timeBegin);
        }

        if ((timeEnd - timeBegin) <= min_time) {
            min_time = (timeEnd - timeBegin);
        }
        imageIndex++;
    }

    UNI_TIME_STATISTICS

    std::cout << "[SUMMARY]:" << std::endl;
    UNI_CI_LOG("avg_time:%fms/image\n", 1.0 * totalTime / imageIndex);
    UNI_CI_LOG("max_time:%fms/image\n", 1.0 * max_time);
    UNI_CI_LOG("min_time:%fms/image\n", 1.0 * min_time);
    return 0;
}
