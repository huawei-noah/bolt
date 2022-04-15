// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef DLLITE_BOLT_H
#define DLLITE_BOLT_H

#include <vector>
#include <string>

namespace bolt {

/** inference pipeline handle */
using ModelHandle = void *;

/** result data memory handle */
using ResultHandle = void *;

/** CPU affinity policy */
enum class AffinityType {
    CPU_HIGH_PERFORMANCE = 0,  ///< performance is high priority(use big core)
    CPU_LOW_POWER = 1,         ///< power is high priority(use small core)
    GPU = 2                    ///< use GPU
};

/** data precision */
enum class TensorType {
    FP32 = 0,   ///< 32 bit float
    FP16 = 1,   ///< 16 bit float
    INT32 = 2,  ///<  32 bit integer
    UINT32 = 3  ///<  32 bit unsigned integer
};

/** multi-dimension data format */
enum class TensorLayout {
    NCHW = 0,       ///< batch->channel->height->width data order
    NHWC = 1,       ///< batch->height->width->channel data order
    NCHWC8 = 2,     ///< batch->channel/8->height->width->8 data order
    ROW_MAJOR = 3,  ///< batch->unit data order
    RNN_MTK = 4     ///< batch->time->unit data order
};

// IOTensor
struct IOTensor {
    std::string name;
    TensorType type;
    TensorLayout layout;
    std::vector<size_t> shape;
    std::pair<void *, size_t> buffer;  // <ptr_to_memory, size of bytes>
};

// For model and algo config, either both use stream (default) or both use path
struct ModelConfig {
    AffinityType affinity;
    std::pair<void *, size_t> modelStream;
    std::pair<void *, size_t> algoStream;
    std::string modelPath;
    std::string algoPath;
};

// Return status
enum class ReturnStatus {
    SUCCESS = 0,  ///< SUCCESS
    FAIL = -1,    ///< FAIL
    NULLPTR = -2  ///< NULLPTR
};

ModelHandle CreateModel(const ModelConfig &modelConfig);

ReturnStatus GetIOFormats(
    ModelHandle modelHandle, std::vector<IOTensor> &inputs, std::vector<IOTensor> &outputs);

ReturnStatus PrepareModel(ModelHandle modelHandle, const std::vector<IOTensor> &inputs);

ReturnStatus GetInputTensors(ModelHandle modelHandle, std::vector<IOTensor> &inputs);

ReturnStatus ResizeInput(ModelHandle modelHandle, const std::vector<IOTensor> &inputs);

ResultHandle AllocResult(ModelHandle modelHandle, const std::vector<IOTensor> &outputs);

ReturnStatus RunModel(
    ModelHandle modelHandle, ResultHandle resultHandle, const std::vector<IOTensor> &inputs);

ReturnStatus GetOutputTensors(ResultHandle resultHandle, std::vector<IOTensor> &outputs);

ReturnStatus FreeResult(ResultHandle resultHandle);

ReturnStatus DestroyModel(ModelHandle modelHandle);

}  // namespace bolt

#endif  // DLLITE_BOLT_H
