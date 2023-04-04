// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "../api/cpp/Bolt.h"
#include "../api/c/bolt.h"
#include "../api/c/bolt_simplify.h"
#include "online_conversion.h"
#include "error.h"
#include "model_common.h"

Bolt::Bolt()
{
    this->boltHandle = nullptr;
    this->resultHandle = nullptr;
    this->inputNum = 0;
    this->outputNum = 0;
    this->inputData = nullptr;
    this->outputData = nullptr;
}

Bolt::~Bolt()
{
    if (this->boltHandle) {
        DestroyModel(this->boltHandle);
        this->boltHandle = nullptr;
    }
    if (this->resultHandle) {
        FreeResultHandle(this->resultHandle);
        this->resultHandle = nullptr;
    }
    if (this->inputNum > 0) {
        FreeTensorDesc(inputNum, inputName, inputN, inputC, inputH, inputW, (DATA_TYPE *)inputDT,
            (DATA_FORMAT *)inputDF);
        this->inputNum = 0;
    }
    if (this->outputNum > 0) {
        FreeTensorDesc(outputNum, outputName, outputN, outputC, outputH, outputW,
            (DATA_TYPE *)outputDT, (DATA_FORMAT *)outputDF);
        this->outputNum = 0;
    }
    if (this->inputData != nullptr) {
        UNI_FREE(inputData);
        this->inputData = nullptr;
    }
    if (this->outputData != nullptr) {
        UNI_FREE(outputData);
        this->outputData = nullptr;
    }
}

std::string Bolt::convert(std::string modelDirectory, std::string modelName, std::string precision)
{
    UNI_DEBUG_LOG("CPP API %s(%s, %s, %s)...\n", __func__, modelDirectory.c_str(),
        modelName.c_str(), precision.c_str());
    std::string boltModelPath = modelDirectory + "/" + modelName + "_";
    if (precision == "FP16") {
        boltModelPath = boltModelPath + "f16.bolt";
    } else if (precision == "FP32") {
        boltModelPath = boltModelPath + "f32.bolt";
    } else {
        UNI_ERROR_LOG("can not convert model to %s.\n", precision.c_str());
        return "";
    }

    ModelSpec *ms = (ModelSpec *)OnlineModelConversion(
        modelDirectory.c_str(), modelName.c_str(), precision.c_str());
    serialize_model_to_file(ms, boltModelPath.c_str());
    OnlineModelReclaim(ms);
    UNI_DEBUG_LOG("CPP API %s(%s) end.\n", __func__, boltModelPath.c_str());
    return boltModelPath;
}

int Bolt::set_num_threads(int threads)
{
    UNI_DEBUG_LOG("CPP API %s(%d).\n", __func__, threads);
    SetNumThreads(threads);
    return 0;
}

int Bolt::load(std::string boltModelPath, std::string affinity)
{
    UNI_DEBUG_LOG("CPP API %s(%s, %s).\n", __func__, boltModelPath.c_str(), affinity.c_str());
    std::map<std::string, AFFINITY_TYPE> m = {
        {"CPU_HIGH_PERFORMANCE", CPU_HIGH_PERFORMANCE},
        {"CPU_LOW_POWER", CPU_LOW_POWER},
        {"CPU", CPU},
        {"GPU", GPU},
        {"XPU", XPU},
    };
    AFFINITY_TYPE affinityType = CPU_HIGH_PERFORMANCE;
    if (m.find(affinity) == m.end()) {
        UNI_ERROR_LOG("can not support affinity:%s.\n", affinity.c_str());
    } else {
        affinityType = m[affinity];
    }
    if (this->boltHandle != nullptr) {
        DestroyModel(this->boltHandle);
    }
    if (this->resultHandle != nullptr) {
        FreeResultHandle(this->resultHandle);
    }
    CreateInference(false, boltModelPath.c_str(), nullptr, affinityType, &boltHandle, &resultHandle);
    if (resultHandle == nullptr || boltHandle == nullptr) {
        return 1;
    }

    FreeTensorDesc(inputNum, inputName, inputN, inputC, inputH, inputW, (DATA_TYPE *)inputDT,
        (DATA_FORMAT *)inputDF);
    FreeTensorDesc(outputNum, outputName, outputN, outputC, outputH, outputW, (DATA_TYPE *)outputDT,
        (DATA_FORMAT *)outputDF);
    CreateInputTensorDesc(boltHandle, &inputNum, &inputName, &inputN, &inputC, &inputH, &inputW,
        (DATA_TYPE **)&inputDT, (DATA_FORMAT **)&inputDF);
    CreateOutputTensorDesc(resultHandle, &outputNum, &outputName, &outputN, &outputC, &outputH,
        &outputW, (DATA_TYPE **)&outputDT, (DATA_FORMAT **)&outputDF);

    if (inputData != nullptr) {
        UNI_FREE(inputData);
    }
    if (outputData != nullptr) {
        UNI_FREE(outputData);
    }
    this->inputData = (void **)UNI_MALLOC(sizeof(void *) * inputNum);
    this->outputData = (void **)UNI_MALLOC(sizeof(void *) * outputNum);
    return 0;
}

std::map<std::string, std::vector<float>> Bolt::infer(
    const std::map<std::string, std::vector<float>> &input)
{
    UNI_DEBUG_LOG("CPP API %s(...)...\n", __func__);
    std::map<std::string, std::vector<float>> res;
    for (int i = 0; i < inputNum; i++) {
        std::string name = inputName[i];
        if (input.find(name) == input.end()) {
            UNI_ERROR_LOG("missed input data %s.\n", name.c_str());
            return res;
        }
        inputData[i] = (void *)((input.find(name))->second).data();
    }
    std::vector<DATA_TYPE> v(inputNum, FP_32);
    RunModelWithType(
        boltHandle, resultHandle, inputNum, (const char **)inputName, v.data(), inputData, NULL);
    GetOutputDataFromResultHandle(resultHandle, outputNum, outputData);
    DATA_TYPE *odts = (DATA_TYPE *)outputDT;
    DATA_FORMAT *odfs = (DATA_FORMAT *)outputDF;
    for (int i = 0; i < outputNum; i++) {
        std::string name = std::string(outputName[i]);
        size_t length = outputN[i] * outputC[i] * outputH[i] * outputW[i];
        std::vector<float> vec(length);
        size_t size = length * GetDataTypeSize(odts[i]);
        if (odfs[i] == NCHWC8 || odfs[i] == NCHWC4 || odts[i] != FP_32) {
            DATA_TYPE odt = odts[i];
            if (odt != FP_32) {
                odt = FP_32;
            }
            if (TransformDataTypeAndFormat(boltHandle, odts[i], odfs[i], outputN[i], outputC[i],
                    outputH[i], outputW[i], outputData[i], odt, NCHW, outputN[i], outputC[i],
                    outputH[i], outputW[i], vec.data()) != 0) {
                UNI_ERROR_LOG("can not transform output %s.\n", name.c_str());
                return res;
            }
        } else {
            UNI_MEMCPY(vec.data(), outputData[i], size);
        }
        res[name] = vec;
    }
    UNI_DEBUG_LOG("CPP API %s(%d)...\n", __func__, (int)res.size());
    return res;
}

std::map<std::string, std::vector<int>> Bolt::get_input_info()
{
    UNI_DEBUG_LOG("CPP API %s()...\n", __func__);
    std::map<std::string, std::vector<int>> res;
    for (int i = 0; i < inputNum; i++) {
        std::vector<int> vec(4);
        vec[0] = inputN[i];
        vec[1] = inputC[i];
        vec[2] = inputH[i];
        vec[3] = inputW[i];
        res[inputName[i]] = vec;
    }
    UNI_DEBUG_LOG("CPP API %s(%d) end.\n", __func__, (int)res.size());
    return res;
}

std::map<std::string, std::vector<int>> Bolt::get_output_info()
{
    UNI_DEBUG_LOG("CPP API %s()...\n", __func__);
    std::map<std::string, std::vector<int>> res;
    for (int i = 0; i < outputNum; i++) {
        std::vector<int> vec(4);
        vec[0] = outputN[i];
        vec[1] = outputC[i];
        vec[2] = outputH[i];
        vec[3] = outputW[i];
        res[outputName[i]] = vec;
    }
    UNI_DEBUG_LOG("CPP API %s(%d) end.\n", __func__, (int)res.size());
    return res;
}
