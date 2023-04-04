// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MODEL_CALIBRATION
#define _H_MODEL_CALIBRATION

#include "inference.hpp"
#include "model_common.h"
#include "data_loader.hpp"
#include "result_format.hpp"
#include "tensor_computing.h"

const int BINS = 2048;
const int NUM_IMAGES_INPUT = 100;

template <typename T>
static EE minmax_value_func(const T *data, I32 len, int mode, F32 *result)
{
    if (len <= 0) {
        return SUCCESS;
    }
    int id = 0;
    EE ret = NOT_SUPPORTED;
    if (mode & 1) {
        T min_s = data[0];
        for (I32 i = 1; i < len; i++) {
            min_s = UNI_MIN(data[i], min_s);
        }
        result[id++] = min_s;
        ret = SUCCESS;
    }
    if (mode & 2) {
        T max_s = data[0];
        for (I32 i = 1; i < len; i++) {
            max_s = UNI_MAX(data[i], max_s);
        }
        result[id++] = max_s;
        ret = SUCCESS;
    }
    return ret;
}

template <typename T>
static void update_histogram(
    const std::vector<U32> &lens, const T *data, int numBins, F32 interval, F32 *histo)
{
    U32 sum = 0;
    for (U32 len : lens) {
        sum += len;
    }
    for (U32 len : lens) {
        std::vector<F32> tmpHis(numBins, 0);
        for (U32 i = 0; i < len; ++i) {
            T tmp = data[i];
            int index = floor(abs(tmp) / interval);
            if (index >= numBins) {
                index = numBins - 1;
            }
            tmpHis[index] += 1;
        }
        for (int i = 0; i < numBins; ++i) {
            histo[i] += tmpHis[i] / (len * lens.size() * 1.0f) * sum;
        }
        data += len;
    }
}

#if defined(_USE_INT8) && !defined(_USE_LITE)
void calibrate_model_with_dataset(std::string dataPath,
    ImageFormat imageFormat,
    DataType inferType,
    F32 scaleValue,
    std::string modelPath,
    ModelSpec *resultMs)
{
    DataType fType;
    if (isQuantMixDataType(inferType)) {
        fType = noQuantDataType(inferType);
    } else {
        UNI_ERROR_LOG("model calibration only support INT8 inference model.\n");
    }

    ModelSpec int8Ms;
    CHECK_STATUS(deserialize_model_from_file(modelPath.c_str(), &int8Ms, inferType));
    CHECK_REQUIREMENT(int8Ms.dt == inferType || int8Ms.dt == fType);
    int8Ms.dt = inferType;

    ModelSpec floatMs;
    CHECK_STATUS(deserialize_model_from_file(modelPath.c_str(), &floatMs, fType));
    floatMs.dt = fType;

    CHECK_STATUS(deserialize_model_from_file(modelPath.c_str(), resultMs, inferType));
    resultMs->dt = inferType;

    U32 elementBytes = bytesOf(fType);
    auto int8CNN = createPipelinefromMs(AFFINITY_CPU_HIGH_PERFORMANCE, &int8Ms, "");
    auto floatCNN = createPipelinefromMs(AFFINITY_CPU_HIGH_PERFORMANCE, &floatMs, "");

    // load calibration data
    std::map<std::string, TensorDesc> originInputDescMap = int8CNN->get_input_desc();
    std::map<std::string, TensorDesc> inputDescMap = originInputDescMap;
    std::vector<TensorDesc> inputDescs;
    for (auto iter : inputDescMap) {
        TensorDesc curDesc = iter.second;
        inputDescs.push_back(curDesc);
    }
    std::vector<std::string> inputPaths;
    std::vector<std::vector<Tensor>> inputs;
    std::map<std::string, std::shared_ptr<U8>> model_tensors_input;

    std::vector<std::string> paths = search_files(dataPath);
    if (paths.size() > 0) {
        if (endswith(paths[0], ".jpg") || endswith(paths[0], ".jpeg")) {
#ifdef _BUILD_EXAMPLE
            inputPaths =
                load_image_with_scale(dataPath, inputDescs, &inputs, imageFormat, scaleValue);
#endif
        } else {
            inputPaths = load_data(dataPath, inputDescs, &inputs);
        }
    } else {
        UNI_ERROR_LOG("The data directory is NULL.");
        return;
    }

    std::vector<U32> calibratedOpIdx;
    std::map<std::string, std::vector<F32>> tensorScale;
    //TODO: add more ops to quantize
    U32 opIdx = int8CNN->find_next_dynamic_scale_op(calibratedOpIdx, 0);

    while (opIdx != 0) {
        auto op = int8CNN->get_operator_by_index(opIdx);
        std::string opName = op->get_name();
        std::string opsName = int8Ms.ops[opIdx].name;
        CHECK_REQUIREMENT(opName == opsName);

        std::vector<std::vector<F32>> scales(
            int8Ms.ops[opIdx].num_inputs + int8Ms.ops[opIdx].num_outputs);
        int8CNN->reready(originInputDescMap);
        auto inputTensors = op->get_input_tensors();
        auto outputTensors = op->get_output_tensors();
        floatCNN->reready(originInputDescMap);
        auto opFloat = floatCNN->get_operator_by_index(opIdx);
        auto floatInputTensors = opFloat->get_input_tensors();
        auto floatOutputTensors = opFloat->get_output_tensors();

        std::vector<std::string> calibrateTensorName;
        std::vector<Tensor> calibrateTensors;
        std::vector<U32> calibrateIdxs;
        std::vector<std::pair<bool, U32>> tensorPosition;

        for (U32 i = 0; i < int8Ms.ops[opIdx].num_inputs; i++) {
            // already set clipping value
            if (inputTensors[i].get_scale() > 0) {
                continue;
            }
            if (int8Ms.ops[opIdx].type == OT_LayerNorm || int8Ms.ops[opIdx].type == OT_Embedding) {
                continue;
            }
            std::string tensorName = int8Ms.ops[opIdx].input_tensors_name[i];
            TensorDesc inDesc = inputTensors[i].get_desc();

            auto it = tensorScale.find(tensorName);
            if (it != tensorScale.end()) {
                scales[i] = tensorScale[tensorName];
                continue;
            }

            // Gets scale from int8 pooling or concat. Label with -1
            if (DT_I8 == inDesc.dt || DT_U8_Q == inDesc.dt) {
                std::vector<F32> scale;
                scale.push_back(-1);
                scales[i] = scale;
                tensorScale[tensorName] = scale;
                continue;
            }

            bool isSoftmaxOutput = false;
            for (int j = opIdx - 1; j >= 0; --j) {
                if (int8Ms.ops[j].num_outputs > 0 &&
                    int8Ms.ops[j].output_tensors_name[0] == tensorName &&
                    int8Ms.ops[j].type == OT_Softmax) {
                    isSoftmaxOutput = true;
                    break;
                }
            }
            if (isSoftmaxOutput) {
                std::vector<F32> scale;
                scale.push_back(-1);  // clipping at 1
                scales[i] = scale;
                tensorScale[tensorName] = scale;
                continue;
            }

            calibrateTensors.push_back(floatInputTensors[i]);
            calibrateTensorName.push_back(tensorName);
            tensorPosition.push_back(std::make_pair(false, i));
            calibrateIdxs.push_back(i);
        }

        for (U32 i = 0; i < int8Ms.ops[opIdx].num_outputs; i++) {
            // already set clipping value
            if (outputTensors[i].get_scale() > 0) {
                continue;
            }
            std::string tensorName = int8Ms.ops[opIdx].output_tensors_name[i];
            TensorDesc desc = outputTensors[i].get_desc();

            auto it = tensorScale.find(tensorName);
            CHECK_REQUIREMENT(it == tensorScale.end());

            if (DT_I8 != desc.dt && DT_U8_Q != desc.dt) {
                continue;
            }

            bool isReluInput = false;
            for (int j = opIdx + 1; j < int8Ms.num_operator_specs; ++j) {
                if (int8Ms.ops[j].num_inputs > 0 &&
                    int8Ms.ops[j].input_tensors_name[0] == tensorName &&
                    int8Ms.ops[j].type == OT_Relu) {
                    isReluInput = true;
                    break;
                }
            }
            if (isReluInput) {
                std::vector<F32> scale;
                scale.push_back(-1);  // clipping at 1
                scales[i + int8Ms.ops[opIdx].num_inputs] = scale;
                tensorScale[tensorName] = scale;
                continue;
            }

            calibrateTensors.push_back(floatOutputTensors[i]);
            tensorPosition.push_back(std::make_pair(true, i));
            calibrateTensorName.push_back(tensorName);
            calibrateIdxs.push_back(i + int8Ms.ops[opIdx].num_inputs);
        }
        if (calibrateTensors.size() == 0) {
            calibratedOpIdx.push_back(opIdx);
            opIdx = int8CNN->find_next_dynamic_scale_op(calibratedOpIdx, opIdx);
            continue;
        }

        for (U32 i = 0; i < calibrateTensors.size(); ++i) {
            std::string tensorName = calibrateTensorName[i];
            Tensor tensor = calibrateTensors[i];
            auto floatDesc = tensor.get_desc();
            U32 dBytes = tensorNumBytes(floatDesc);

            U32 useSplitNum = (dBytes > 1e7) ? 8 : NUM_IMAGES_INPUT;
            U8 *dBuf = (U8 *)malloc(dBytes * useSplitNum);
            U8 *d = dBuf;
            std::vector<F32> histogram;
            std::vector<U32> tensorSize;
            F32 lastMaxAbs = -1;
            F32 interval = 0;
            F32 maxAbs = 0;
            F32 minmax[2] = {1, -1};

            for (U32 j = 0; j < inputs.size(); j++) {
                U32 index = 0;
                for (auto &iter : inputDescMap) {
                    model_tensors_input[iter.first] =
                        ((CpuMemory *)inputs[j][index].get_memory())->get_shared_ptr();
                    iter.second = inputs[j][index].get_desc();
                    index++;
                }
                floatCNN->reready(inputDescMap);
                floatCNN->set_input_by_assign(model_tensors_input);

                floatCNN->run_till_breakpoint(opIdx);
                auto resizedTensors = tensorPosition[i].first ? opFloat->get_output_tensors()
                                                              : opFloat->get_input_tensors();
                tensorSize.push_back(
                    tensorNumElements(resizedTensors[tensorPosition[i].second].get_desc()));
                dBytes = tensorSize.back() * elementBytes;
                UNI_MEMCPY(d,
                    ((CpuMemory *)(resizedTensors[tensorPosition[i].second].get_memory()))->get_ptr(),
                    dBytes);

                CHECK_STATUS(minmax_value_func<F32>((F32 *)d, tensorSize.back(), 3, minmax));
                maxAbs = UNI_MAX(maxAbs, UNI_MAX(UNI_ABS(minmax[0]), UNI_ABS(minmax[1])));

                d += dBytes;

                if (((j + 1) % useSplitNum == 0) || ((j + 1) == inputs.size())) {
                    F32 *ptr_d = (F32 *)dBuf;

                    if ((j + 1) > useSplitNum) {
                        if (maxAbs <= lastMaxAbs) {
                            interval = lastMaxAbs / BINS;
                        } else {
                            F32 numPerBin = (F32)maxAbs / lastMaxAbs;
                            //lastMaxAbs = max; -> may optimize accuracy.
                            histogram = compress_histogram(histogram, numPerBin, lastMaxAbs);
                            interval = maxAbs / BINS;
                        }
                    } else {
                        interval = maxAbs / BINS;
                        histogram.resize(BINS, 0);
                    }

                    update_histogram(tensorSize, ptr_d, BINS, interval, histogram.data());
                    lastMaxAbs = maxAbs;
                    d = dBuf;
                    tensorSize.clear();
                }
            }
            free(dBuf);
            std::vector<F32> scale = compute_scale_with_KL(histogram, interval);
            scales[calibrateIdxs[i]] = scale;
            tensorScale[tensorName] = scale;
        }

        if (int8Ms.ops[opIdx].num_quant_feature == 1 &&
            -2 == int8Ms.ops[opIdx].feature_scale[0].scale[0]) {
            std::vector<F32> outputScale;
            outputScale.push_back(-2);
            scales.back() = outputScale;
        }

        op->set_feature_scale(scales);

        // Store scales into result model
        // Could be labelled with -2
        if (nullptr != resultMs->ops[opIdx].feature_scale) {
            for (U32 i = 0; i < resultMs->ops[opIdx].num_quant_feature; i++) {
                if (nullptr != resultMs->ops[opIdx].feature_scale[i].scale) {
                    delete[] resultMs->ops[opIdx].feature_scale[i].scale;
                }
            }
            delete[] resultMs->ops[opIdx].feature_scale;
        }

        resultMs->ops[opIdx].num_quant_feature = scales.size();
        resultMs->ops[opIdx].feature_scale =
            (QuantSpec *)mt_malloc(scales.size() * sizeof(QuantSpec));

        for (U32 i = 0; i < scales.size(); i++) {
            resultMs->ops[opIdx].feature_scale[i].num_scale = scales[i].size();
            U32 scaleBytes = scales[i].size() * sizeof(F32);
            resultMs->ops[opIdx].feature_scale[i].scale = (F32 *)mt_malloc(scaleBytes);
            UNI_MEMCPY(resultMs->ops[opIdx].feature_scale[i].scale, scales[i].data(), scaleBytes);
        }

        calibratedOpIdx.push_back(opIdx);
        opIdx = int8CNN->find_next_dynamic_scale_op(calibratedOpIdx, opIdx);
    }

    CHECK_STATUS(mt_destroy_model(&int8Ms));
    CHECK_STATUS(mt_destroy_model(&floatMs));
}
#endif
#endif
