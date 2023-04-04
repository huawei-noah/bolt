// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "tensor_computing.h"
#include <random>
#include <chrono>

EE random_infer_output_size(
    Tensor *inputTensor, RandomParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo)
{
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc desc;
    if (inputTensor != nullptr) {
        desc = inputTensor->get_desc();
    } else {
        desc.dt = p.dt;
        desc.nDims = p.num_shape;
        for (int i = 0; i < p.num_shape; i++) {
            desc.dims[i] = p.shape[p.num_shape - 1 - i];
        }
    }
    outputTensor->resize(desc);
    return SUCCESS;
}

template <typename T>
static void generate(RandomParamSpec p, T *data, size_t length)
{
    if (p.seed == UNI_RESERVE) {
        p.seed = std::chrono::system_clock::now().time_since_epoch().count();
    }
    if (p.mode == RANDOM_NORMAL) {
        std::default_random_engine generator{static_cast<U32>(p.seed)};
        std::normal_distribution<float> distribution{p.value[0], p.value[1]};
        for (size_t i = 0; i < length; i++) {
            data[i] = distribution(generator);
        }
    } else {
        std::default_random_engine generator{static_cast<U32>(p.seed)};
        std::uniform_real_distribution<double> distribution{p.value[1], p.value[0]};
        for (size_t i = 0; i < length; i++) {
            data[i] = distribution(generator);
        }
    }
}

EE random(RandomParamSpec p, Tensor outputTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc outputDesc = outputTensor.get_desc();
    size_t length = tensorNumElements(outputDesc);
    if (length == 0) {
        return SUCCESS;
    }
    void *output = get_ptr_from_tensor(outputTensor, arch);
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
        ret = SUCCESS;
        if (outputDesc.dt == DT_F32) {
            generate<F32>(p, (F32 *)output, length);
#ifdef _USE_FP16
        } else if (outputDesc.dt == DT_F16) {
            generate<F16>(p, (F16 *)output, length);
#endif
        } else {
            ret = NOT_SUPPORTED;
        }
    }
    return ret;
}
