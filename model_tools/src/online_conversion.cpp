// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "online_conversion.h"
#include "model_converter.h"
#include "model_optimizer.hpp"
#include "model_data_type_converter.h"

bool fileExist(const std::string &name)
{
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

void *OnlineModelConversion(const char *storagePath,
    const char *modelName,
    const char *inferPrecision,
    I32 removeProcessOpsNum)
{
    DataConvertType converterMode = F32_to_F32;
    if (inferPrecision == std::string("PTQ")) {
        converterMode = F32_to_F32;
    } else if (inferPrecision == std::string("FP16")) {
        converterMode = F32_to_F16;
    } else if (inferPrecision == std::string("FP32")) {
        converterMode = F32_to_F32;
    } else {
        UNI_ERROR_LOG("Unknown converter data precision: %s.\n", inferPrecision);
        exit(1);
    }

    ModelSpec *originalMs = new ModelSpec();
    ModelSpec *targetMs = new ModelSpec();
    CHECK_STATUS(mt_create_model(originalMs));
    CHECK_STATUS(mt_create_model(targetMs));

    std::string spStr = storagePath;
    std::string mnStr = modelName;
    std::string prefix = (spStr.at(spStr.size() - 1) == '/') ? (spStr + mnStr)
                                                             : (spStr + "/" + mnStr);
    if (0) {
#ifdef _USE_CAFFE
    } else if (fileExist(prefix + ".prototxt") && fileExist(prefix + ".caffemodel")) {
        UNI_INFO_LOG("Start to convert %s.caffemodel...\n", prefix.c_str());
        caffe_converter(storagePath, modelName, originalMs);
#endif
#ifdef _USE_ONNX
    } else if (fileExist(prefix + ".onnx")) {
        UNI_INFO_LOG("Start to convert %s.onnx...\n", prefix.c_str());
        onnx_converter(storagePath, modelName, removeProcessOpsNum, originalMs);
#endif
#ifdef _USE_TFLITE
    } else if (fileExist(prefix + ".tflite")) {
        UNI_INFO_LOG("Start to convert %s.tflite...\n", prefix.c_str());
        tflite_converter(storagePath, modelName, originalMs);
#endif
#ifdef _USE_TENSORFLOW
    } else if (fileExist(prefix + ".json")) {
        UNI_INFO_LOG("Start to convert %s.json...\n", prefix.c_str());
        tensorflow_converter(storagePath, modelName, originalMs);
#endif
    } else {
        UNI_ERROR_LOG("Can not find %s.prototxt/caffemodel, %s.onnx, %s.tflite or %s.json model "
                      "file.\n",
            prefix.c_str(), prefix.c_str(), prefix.c_str(), prefix.c_str());
        exit(1);
    }

    UNI_DEBUG_LOG("Start to optimize graph...\n");
    ModelSpecOptimizer msOptimizer;
    msOptimizer.suggest(inferPrecision == std::string("PTQ"));
    msOptimizer.optimize(originalMs);

    CHECK_STATUS(ms_datatype_converter(originalMs, targetMs, converterMode, "NOQUANT"));
    CHECK_STATUS(mt_destroy_model(originalMs));
    delete originalMs;
    //operator_relationship(targetMs);
    return (void *)targetMs;
}

void OnlineModelReclaim(void *ms)
{
    ModelSpec *targetMs = (ModelSpec *)ms;
    CHECK_STATUS(mt_destroy_model(targetMs));
    delete targetMs;
    return;
}
