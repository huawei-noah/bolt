// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TENSOR_COMPUTING_FP16
#define _H_TENSOR_COMPUTING_FP16

#include "tensor_desc.h"
#include "parameter_spec.h"
#include "gcl.h"
#include "ocl_desc_trans.h"
#define BUFFER_ALIGN_BASE 128
#define CHECK_MEET_IMAGE_LIMITS(width, height, depth) \
    (gcl_check_meet_device_image3d_limits(            \
        OCLContext::getInstance().handle.get(), width, height, depth))

inline std::vector<I32> build_conv_forward_algorithm_flag(TensorDesc inputDesc,
    std::vector<TensorDesc> filterDesc,
    OperatorType opType,
    GCLMemType imt,
    GCLMemType omt,
    ConvolutionParamSpec convParamSpec)
{
    std::vector<I32> flag;
    flag.push_back(opType);
    flag.push_back(convParamSpec.convolution_type);
    for (U32 i = 0; i < inputDesc.nDims; i++) {
        flag.push_back(inputDesc.dims[i]);
    }
    for (auto &p : filterDesc) {
        for (U32 i = 0; i < p.nDims; i++) {
            flag.push_back(p.dims[i]);
        }
    }
    flag.push_back(convParamSpec.kernel_t);
    flag.push_back(convParamSpec.kernel_h);
    flag.push_back(convParamSpec.kernel_w);
    flag.push_back(convParamSpec.stride_t);
    flag.push_back(convParamSpec.stride_h);
    flag.push_back(convParamSpec.stride_w);
    flag.push_back(convParamSpec.group);
    flag.push_back(convParamSpec.dilatedRate_t);
    flag.push_back(convParamSpec.dilatedRate_h);
    flag.push_back(convParamSpec.dilatedRate_w);
    flag.push_back(imt);
    flag.push_back(omt);
    return flag;
}

inline std::vector<I32> build_fully_connected_forward_algorithm_flag(
    TensorDesc inputDesc, TensorDesc filterDesc, GCLMemType imt, GCLMemType omt)
{
    std::vector<I32> flag;
    flag.push_back(OT_FC);
    for (U32 i = 0; i < inputDesc.nDims; i++) {
        flag.push_back(inputDesc.dims[i]);
    }
    for (U32 i = 0; i < filterDesc.nDims; i++) {
        flag.push_back(filterDesc.dims[i]);
    }
    flag.push_back(imt);
    flag.push_back(omt);
    return flag;
}

inline std::vector<I32> build_matmul_forward_algorithm_flag(TensorDesc matrixADesc,
    bool transposeA,
    TensorDesc matrixBDesc,
    bool transposeB,
    GCLMemType amt,
    GCLMemType bmt,
    GCLMemType cmt)
{
    std::vector<I32> flag;
    flag.push_back(OT_MatMul);
    flag.push_back(transposeA);
    flag.push_back(transposeB);
    for (U32 i = 0; i < matrixADesc.nDims; i++) {
        flag.push_back(matrixADesc.dims[i]);
    }
    for (U32 i = 0; i < matrixBDesc.nDims; i++) {
        flag.push_back(matrixBDesc.dims[i]);
    }
    flag.push_back(amt);
    flag.push_back(bmt);
    flag.push_back(cmt);
    return flag;
}

inline std::vector<I32> build_rnn_forward_algorithm_flag(
    TensorDesc inputDesc, std::vector<TensorDesc> filterDesc, RNNParamSpec rnnPara)
{
    std::vector<I32> flag;
    flag.push_back(OT_RNN);
    flag.push_back(rnnPara.steps);
    flag.push_back(rnnPara.mode);
    flag.push_back(rnnPara.num_outputs);
    flag.push_back(rnnPara.num_projection);
    flag.push_back(rnnPara.bi_direction);
    for (U32 i = 0; i < inputDesc.nDims; i++) {
        flag.push_back(inputDesc.dims[i]);
    }
    for (auto &p : filterDesc) {
        for (U32 i = 0; i < p.nDims; i++) {
            flag.push_back(p.dims[i]);
        }
    }
    return flag;
}
#endif
