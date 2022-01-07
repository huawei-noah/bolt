// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _MATMUL_MALI_FP16
#define _MATMUL_MALI_FP16

#include "gpu/mali/fp16/tensor_computing_fp16.h"

inline EE trans_matmul_input(GCLHandle_t handle,
    GCLMem_t in,
    GCLMem_t out,
    DataType dt,
    U32 w,
    U32 h,
    U32 c,
    bool setKernelVec = true)
{
    U32 iw_str, ih_str, iw_off, ih_off;
    U32 ow_str, oh_str, ow_off, oh_off;
    CHECK_STATUS(gclmem_get_desc_padding(in->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off));
    CHECK_STATUS(gclmem_get_desc_padding(out->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off));
    U32 i_off = ih_off * iw_str + iw_off;
    U32 o_off = oh_off * ow_str + ow_off;

    char kernelName[128];
    Kernel kernel;
    KernelOpt kernelOpt;
    U32 gs[3] = {(w + 3) >> 2, (h + 3) >> 2, c};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    CHECK_STATUS(set_common_opt(
        dt, in->desc.memType, out->desc.memType, "matmul_trans_input", kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, ow_str, oh_str, i_off, o_off, w, h,
        gs[0], gs[1], in->mem, out->mem));
    if (setKernelVec) {
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
    } else {
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    }
    return SUCCESS;
}

inline void get_reshaped_desc(TensorDesc matrixADesc,
    TensorDesc matrixBDesc,
    bool transposeA,
    bool transposeB,
    bool *needReshapeA,
    bool *needReshapeB,
    TensorDesc *matrixADescReshaped,
    TensorDesc *matrixBDescReshaped)
{
    TensorDesc descA = matrixADesc;
    TensorDesc descB = matrixBDesc;
    U32 nDimsA = descA.nDims;
    U32 nDimsB = descB.nDims;
    if (transposeA && nDimsA >= 2) {  //if A is T, set it to N, for put reduce dim on dims[0]
        U32 i = descA.dims[0];
        descA.dims[0] = descA.dims[1];
        descA.dims[1] = i;
    }
    if (!transposeB && nDimsB >= 2) {  //if B is N, set it to T ,for put reduce dim on dims[0]
        U32 i = descB.dims[0];
        descB.dims[0] = descB.dims[1];
        descB.dims[1] = i;
    }
    U32 nDimsMax = (nDimsA >= nDimsB) ? nDimsA : nDimsB;
    *needReshapeA = false;
    *needReshapeB = false;
    for (U32 i = 2; i < nDimsMax; i++) {
        U32 dimA = (i < nDimsA) ? descA.dims[i] : 1;
        U32 dimB = (i < nDimsB) ? descB.dims[i] : 1;
        if (dimA > dimB) {
            if (dimB != 1) {
                CHECK_STATUS(NOT_MATCH);
            }
            descA.dims[1] = descA.dims[1] * dimA;
            descA.dims[i] = 1;
            *needReshapeA = true;
        }
        if (dimA < dimB) {
            if (dimA != 1) {
                CHECK_STATUS(NOT_MATCH);
            }
            descB.dims[1] = descB.dims[1] * dimB;
            descB.dims[i] = 1;
            *needReshapeB = true;
        }
    }
    if (matrixADescReshaped) {
        *matrixADescReshaped = descA;
    }
    if (matrixBDescReshaped) {
        *matrixBDescReshaped = descB;
    }
}

inline bool need_process_matmul_input(
    TensorDesc desc, GCLMemType mt, bool needReshape, bool curTrans, bool targetTrans)
{
    bool useImg = check_qualcomm_device();
    if (desc.df == DF_NCHWC4) {
        return true;
    }
    if (needReshape) {
        return true;
    }
    if (curTrans != targetTrans) {
        return true;
    };
    if (useImg && mt == GCL_MEM_BUF) {
        return true;
    }
    return false;
}

EE matmul_infer_forward_tmp_bytes_mali_fp16(TensorDesc matrixADesc,
    bool transposeA,
    TensorDesc matrixBDesc,
    bool transposeB,
    TensorDesc matrixCDesc,
    GCLMemDesc gclmemMatrixADesc,
    GCLMemDesc gclmemMatrixBDesc,
    GCLMemDesc gclmemMatrixCDesc,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo);

EE matmul_mali_fp16(GCLHandle_t handle,
    TensorDesc matrixADesc,
    bool transposeA,
    GCLMem_t matrixA,
    TensorDesc matrixBDesc,
    bool transposeB,
    GCLMem_t matrixB,
    TensorDesc biasDesc,
    GCLMem_t bias,
    std::vector<GCLMem_t> tmp,
    TensorDesc matrixCDesc,
    GCLMem_t matrixC,
    ForwardRunInfoMali_t forwardRunInfo);
#endif
