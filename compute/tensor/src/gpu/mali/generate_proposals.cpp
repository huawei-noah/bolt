// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "sys.h"
#include "tensor_desc.h"
#include "error.h"
#include "gpu/mali/tensor_computing_mali.h"
#include "gpu/mali/fp16/generate_proposals_mali_fp16.h"

inline EE generate_proposals_checkpara_mali(GCLHandle_t handle,
    TensorDesc deltaDesc,
    GCLMem_t delta,
    TensorDesc logitDesc,
    GCLMem_t logit,
    TensorDesc imgInfoDesc,
    GCLMem_t imgInfo,
    TensorDesc anchorDesc,
    GCLMem_t anchor,
    GenerateProposalsParamSpec generateProposalsParam,
    GCLMem_t tmpBuf,
    U8 *tmpCpu,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    if (handle == nullptr || delta == nullptr || logit == nullptr || imgInfo == nullptr ||
        anchor == nullptr || tmpBuf == nullptr || tmpCpu == nullptr || output == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (deltaDesc.nDims < 4 || deltaDesc.nDims != logitDesc.nDims) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (deltaDesc.dims[0] != logitDesc.dims[0] || deltaDesc.dims[1] != logitDesc.dims[1] ||
        deltaDesc.dims[2] != logitDesc.dims[2] * 4 || deltaDesc.dims[3] != logitDesc.dims[3]) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (tensorNumElements(anchorDesc) != deltaDesc.dims[2]) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (tensorNumElements(imgInfoDesc) < 2) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (outputDesc.dims[0] != 4) {
        CHECK_STATUS(NOT_MATCH);
    }
    U32 proposalLen = tensorNumElements(logitDesc);
    U32 preNmsTop = generateProposalsParam.pre_nms_topN;
    U32 postNmsTop = generateProposalsParam.post_nms_topN;
    if (preNmsTop > proposalLen) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (postNmsTop > preNmsTop) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (outputDesc.dims[1] != postNmsTop) {
        CHECK_STATUS(NOT_MATCH);
    }
    return SUCCESS;
}

EE generate_proposals_infer_forward_tmp_bytes_mali(TensorDesc deltaDesc,
    TensorDesc logitDesc,
    GCLMemDesc gclMemLogitDesc,
    GenerateProposalsParamSpec generateProposalsParam,
    U32 *bytes)
{
    return generate_proposals_infer_forward_tmp_bytes_mali_fp16(
        deltaDesc, logitDesc, gclMemLogitDesc, generateProposalsParam, bytes);
}

EE generate_proposals_mali(GCLHandle_t handle,
    TensorDesc deltaDesc,
    GCLMem_t delta,
    TensorDesc logitDesc,
    GCLMem_t logit,
    TensorDesc imgInfoDesc,
    GCLMem_t imgInfo,
    TensorDesc anchorDesc,
    GCLMem_t anchor,
    GenerateProposalsParamSpec generateProposalsParam,
    GCLMem_t tmpBuf,
    U8 *tmpCpu,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(generate_proposals_checkpara_mali(handle, deltaDesc, delta, logitDesc, logit,
        imgInfoDesc, imgInfo, anchorDesc, anchor, generateProposalsParam, tmpBuf, tmpCpu,
        outputDesc, output));
    return generate_proposals_mali_fp16(handle, deltaDesc, delta, logitDesc, logit, imgInfoDesc,
        imgInfo, anchorDesc, anchor, generateProposalsParam, tmpBuf, tmpCpu, outputDesc, output);
}
