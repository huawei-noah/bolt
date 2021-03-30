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

std::map<std::string, U8 *> transform_input(
    std::map<std::string, TensorDesc> inputDescs, std::vector<Tensor> image, U8 *inputBinPtr)
{
    std::map<std::string, U8 *> input;
    int index = 0;
    for (auto iter : inputDescs) {
        U8 *inputPtr;
        if (inputBinPtr) {
            inputPtr = inputBinPtr;
        } else {
            inputPtr = (U8 *)((CpuMemory *)image[index].get_memory())->get_ptr();
        }
        input[iter.first] = inputPtr;
        index++;
    }
    return input;
}

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
    int topK = 5;
    int category = -1;
    int loopTime = 1;
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
    if (parse_res.topK.second) {
        topK = parse_res.topK.first;
    }
    if (parse_res.loopTime.second) {
        loopTime = parse_res.loopTime.first;
    }
    if (parse_res.correctLable.second) {
        category = parse_res.correctLable.first;
    }
    char *inputBinName = nullptr;
    if (parse_res.readInputBinName.second) {
        inputBinName = parse_res.readInputBinName.first;
    }
    char *outputBinName = nullptr;
    if (parse_res.writeOutputBinName.second) {
        outputBinName = parse_res.writeOutputBinName.first;
    }

    double timeBegin = ut_time_ms();
    auto cnn = createPipeline(affinityPolicyName, modelPath, algorithmMapPath);
    double timeEnd = ut_time_ms();
    std::cout << "Prepare time = " << timeEnd - timeBegin << "ms" << std::endl;

    // load images
    std::map<std::string, TensorDesc> inputDescs = cnn->get_input_desc();
    std::vector<std::vector<Tensor>> images;
    std::vector<TensorDesc> imageDescs;
    std::vector<std::string> imagePaths;
    U8 *inputBinPtr = nullptr;
    for (auto iter : inputDescs) {
        imageDescs.push_back(iter.second);
    }
#ifdef _USE_FP16
    if (parse_res.readInputBinName.second) {
        U32 size = getBinFileSize(imageDir, inputBinName);
        inputBinPtr = new U8[size];
        readF32BinToF16((F16 *)inputBinPtr, size / bytesOf(DT_F32), imageDir, inputBinName);
    } else {
#endif
        imagePaths = load_image_with_scale(imageDir, imageDescs, &images, imageFormat, scaleValue);
#ifdef _USE_FP16
    }
#endif

    std::map<int, int> categoryNum;
    double totalTime = 0;
    double max_time = -DBL_MAX;
    double min_time = DBL_MAX;
    U32 imageIndex = 0;
    std::cout << "[RESULT]:" << std::endl;
    int top1Index = 0;
    int top1Match = 0;
    int topKMatch = 0;
    UNI_INFO_LOG("WARM UP\n");
    for (int i = 0; i < 2; i++) {
        if (images.size() > 0 || inputBinPtr != nullptr) {
            std::map<std::string, U8 *> input = transform_input(inputDescs, images[0], inputBinPtr);
            cnn->set_input_by_copy(input);
        }
        cnn->run();
    }
#ifdef _USE_MALI
    if (strcmp(affinityPolicyName, "GPU") == 0) {
        gcl_finish(OCLContext::getInstance().handle.get());
    }
#endif
    UNI_INFO_LOG("RUN\n");
    for (imageIndex = 0; imageIndex < imagePaths.size();) {
        // stage3: set input
        std::map<std::string, std::shared_ptr<Tensor>> outMap;
        double loop_max_time = -DBL_MAX;
        double loop_min_time = DBL_MAX;
        double loop_total_time = 0;

        U8 *res = nullptr;
        TensorDesc resDesc;
        std::cout << imagePaths[imageIndex] << " : " << std::endl;
        std::map<std::string, U8 *> input =
            transform_input(inputDescs, images[imageIndex], inputBinPtr);

        for (int i = 0; i < loopTime; i++) {
            timeBegin = ut_time_ms();
            cnn->set_input_by_copy(input);
            // stage4: run
            cnn->run();

            // stage5: process result
            outMap = cnn->get_output();
            Tensor result = *(outMap.begin()->second);
            auto mem = result.get_memory();
            if (mem->get_mem_type() == OCLMem) {
#ifdef _USE_MALI
                res = (U8 *)((OclMemory *)mem)->get_mapped_ptr();
#endif
            } else {
                res = (U8 *)((CpuMemory *)mem)->get_ptr();
            }
            resDesc = result.get_desc();
            timeEnd = ut_time_ms();
            double time = (timeEnd - timeBegin);
            loop_total_time += time;
            if (time < loop_min_time) {
                loop_min_time = time;
            }
            if (time > loop_max_time) {
                loop_max_time = time;
            }
        }
        totalTime += (loop_total_time) / loopTime;
        if (loopTime > 1) {
            UNI_CI_LOG("loop %d times for set_input + run + get_output\n", loopTime);
            UNI_CI_LOG("avg_time:%fms/loop\n", loop_total_time / loopTime);
            UNI_CI_LOG("max_time:%fms/loop\n", loop_max_time);
            UNI_CI_LOG("min_time:%fms/loop\n", loop_min_time);
        }

#ifdef _USE_FP16
        if (parse_res.writeOutputBinName.second) {
            U32 num = tensorNumElements(resDesc);
            CI8 *dataName = outputBinName;
            writeF16ToF32Bin((F16 *)res, num, imageDir, dataName);
        }
#endif

        std::vector<int> topKResult = topK_index(res, resDesc, topK);
        top1Index = topKResult[0];
        if (category != -1) {
            if (top1Index == category) {
                top1Match++;
            }
            for (int i = 0; i < topK; i++) {
                if (topKResult[i] == category) {
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
        if (categoryNum.count(top1Index) == 0) {
            categoryNum[top1Index] = 1;
        } else {
            categoryNum[top1Index] = categoryNum[top1Index] + 1;
        }
        imageIndex++;
    }
    imageIndex = UNI_MAX(imageIndex, 1);
    cnn->saveAlgorithmMapToFile(algorithmMapPath);

    UNI_TIME_STATISTICS

    std::cout << "[CATEGORY]:" << std::endl;
    std::cout << "category\tnum" << std::endl;
    for (auto elem : categoryNum) {
        std::cout << elem.first << "\t" << elem.second << std::endl;
    }
    std::cout << "[SUMMARY]:" << std::endl;
    UNI_CI_LOG("top1:%f\n", 1.0 * top1Match / imageIndex);
    UNI_CI_LOG("top%d:%f\n", topK, 1.0 * topKMatch / imageIndex);
    UNI_CI_LOG("avg_time:%fms/image\n", 1.0 * totalTime / imageIndex);
    UNI_CI_LOG("max_time:%fms/image\n", 1.0 * max_time);
    UNI_CI_LOG("min_time:%fms/image\n", 1.0 * min_time);
    if (inputBinPtr) {
        delete[] inputBinPtr;
    }
    return 0;
}
