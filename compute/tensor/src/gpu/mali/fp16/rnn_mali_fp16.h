// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _RNN_MALI_FP16
#define _RNN_MALI_FP16

#include "gpu/mali/fp16/tensor_computing_fp16.h"

inline bool needReshapeInput(GCLMemDesc desc)
{
    bool needReshape = false;
    if (desc.memFormat == DF_NCHWC4) {
        needReshape = true;
    } else {
        U32 iw_str, ih_str;
        gclmem_get_desc_padding(desc, &iw_str, &ih_str, NULL, NULL, NULL);
        if (iw_str != desc.dims[0] || ih_str != desc.dims[1]) {
            needReshape = true;
        }
        if (desc.memType != GCL_MEM_BUF) {
            for (U32 i = 0; i < desc.nDims - 3; i++) {
                if (desc.dims[i] > 1) {
                    needReshape = true;
                    break;
                }
            }
            if (desc.dims[desc.nDims - 1] > 1) {
                needReshape = true;
            }
        }
    }
    return needReshape;
}

EE rnn_transform_filter_bytes_mali_fp16(TensorDesc filterDesc,
    RNNParamSpec rnnParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc *ftmDesc);

EE rnn_transform_filter_mali_fp16(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    GCLMem_t tmpBuf,
    RNNParamSpec rnnPara,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem,
    ForwardRunInfoMali_t forwardRunInfo);

EE rnn_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    GCLMemDesc gclmemInputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    RNNParamSpec rnnPara,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo);

EE rnn_mali_fp16(GCLHandle_t handle,
    std::vector<TensorDesc> inputDescs,
    GCLMem_t input,
    std::vector<TensorDesc> filterDescs,
    GCLMem_t filter,
    std::vector<TensorDesc> biasDescs,
    GCLMem_t bias,
    RNNParamSpec rnnPara,
    std::vector<GCLMem_t> tmp,
    std::vector<TensorDesc> outputDescs,
    GCLMem_t output,
    ForwardRunInfoMali_t forwardRunInfo);
#endif
