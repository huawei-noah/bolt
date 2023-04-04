// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/matmul_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/gemm_tn_opt.h"
#include "gpu/mali/cl/kernel_option/transpose_opt.h"

inline EE matmul_checkpara_mali_fp16(
    TensorDesc matrixADesc, TensorDesc matrixBDesc, TensorDesc matrixCDesc)
{
    if (matrixADesc.dt != matrixBDesc.dt || matrixADesc.dt != matrixCDesc.dt) {
        return NOT_MATCH;
    }
    return SUCCESS;
}

inline void get_desc_whc(TensorDesc desc, U32 *c, U32 *h, U32 *w)
{
    U32 nDims = desc.nDims;
    U32 wv = desc.dims[0];
    U32 hv = desc.dims[1];
    U32 cv = 1;
    for (U32 i = 2; i < nDims; i++) {
        cv *= desc.dims[i];
    }
    *w = wv;
    *h = hv;
    *c = cv;
}

inline void get_process_matmul_input_tmp_bytes(TensorDesc desc,
    TensorDesc descReshaped,
    GCLMemDesc gclmemDesc,
    bool needReshape,
    bool curTrans,
    bool targetTrans,
    U32 alignVal,
    U32 *tmpBufSize,
    U32 *tmpImgSize)
{
    OclMemory tmpOclMem;
    OclMemoryImg tmpOclImg;
    OclMemory *memPtr;
    bool useImg;
    if (desc.df == DF_NCHWC4) {
        desc.df = DF_NCHW;
        if (curTrans == targetTrans && !needReshape) {
            useImg = check_qualcomm_device();
            if (useImg) {
                memPtr = (OclMemory *)(&tmpOclImg);
                memPtr->resize(desc);
                U32 str[3];
                memPtr->stride(str);
                if (CHECK_MEET_IMAGE_LIMITS(str[0], str[1], str[2])) {
                    tmpImgSize[0] = str[0];
                    tmpImgSize[1] = str[1];
                    tmpImgSize[2] = str[2];
                } else {
                    useImg = false;
                }
            }
            if (!useImg) {
                memPtr = &tmpOclMem;
                memPtr->resize(desc);
                U32 w = desc.dims[0];
                U32 pr = UNI_ALIGN(w, alignVal) - w;
                memPtr->padding(0, pr, 0, 0);
                U32 bytes = memPtr->bytes();
                (*tmpBufSize) += UNI_ALIGN(bytes, BUFFER_ALIGN_BASE);
            }
            return;
        }
        memPtr = &tmpOclMem;
        memPtr->resize(desc);
        U32 bytes = memPtr->bytes();
        (*tmpBufSize) += UNI_ALIGN(bytes, BUFFER_ALIGN_BASE);
    }

    if (needReshape) {
        if (curTrans == targetTrans) {
            U32 w, h, c;
            get_desc_whc(desc, &c, &h, &w);
            desc.dims[0] = h;
            desc.dims[1] = w;
            tmpOclMem.resize(desc);
            U32 bytes = tmpOclMem.bytes();
            (*tmpBufSize) += UNI_ALIGN(bytes, BUFFER_ALIGN_BASE);
            curTrans = !curTrans;
        } else if (gclmemDesc.num != tensorNumElements(desc)) {
            tmpOclMem.resize(desc);
            U32 bytes = tmpOclMem.bytes();
            (*tmpBufSize) += UNI_ALIGN(bytes, BUFFER_ALIGN_BASE);
        }
        desc = descReshaped;
    }

    useImg = check_qualcomm_device();
    if (curTrans == targetTrans) {
        if (useImg && gclmemDesc.memType == GCL_MEM_BUF) {
            tmpOclImg.resize(desc);
            U32 str[3];
            tmpOclImg.stride(str);
            if (CHECK_MEET_IMAGE_LIMITS(str[0], str[1], str[2])) {
                tmpImgSize[0] = str[0];
                tmpImgSize[1] = str[1];
                tmpImgSize[2] = str[2];
            }
        } else {
            CHECK_STATUS(NOT_MATCH);
        }
    } else {
        U32 w, h, c;
        get_desc_whc(desc, &c, &h, &w);
        desc.dims[0] = h;
        desc.dims[1] = w;
        if (useImg) {
            memPtr = &tmpOclImg;
            memPtr->resize(desc);
            U32 str[3];
            memPtr->stride(str);
            if (CHECK_MEET_IMAGE_LIMITS(str[0], str[1], str[2])) {
                tmpImgSize[0] = str[0];
                tmpImgSize[1] = str[1];
                tmpImgSize[2] = str[2];
            } else {
                useImg = false;
            }
        }
        if (!useImg) {
            memPtr = &tmpOclMem;
            memPtr->resize(desc);
            U32 pr = UNI_ALIGN(h, alignVal) - h;
            memPtr->padding(0, pr, 0, 0);
            U32 bytes = memPtr->bytes();
            (*tmpBufSize) += UNI_ALIGN(bytes, BUFFER_ALIGN_BASE);
        }
    }
}

inline EE process_matmul_input(GCLHandle_t handle,
    TensorDesc desc,
    TensorDesc descReshaped,
    GCLMem_t matrix,
    bool needReshape,
    bool curTrans,
    bool targetTrans,
    U32 alignVal,
    U32 *tmpOff,
    GCLMem_t tmpbuf,
    GCLMem_t tmpImg,
    TensorDesc *descTran,
    GCLMem_t matrixTran)
{
    OclMemory tmpOclMem;
    OclMemoryImg tmpOclImg;
    OclMemory *memPtr;
    GCLMem curMatrix = *matrix;
    GCLMem tmpMatrix;
    DataType dt = desc.dt;
    if (desc.df == DF_NCHWC4) {
        desc.df = DF_NCHW;
        memPtr = (OclMemory *)(&tmpOclMem);
        if (curTrans == targetTrans && !needReshape) {
            if (tmpImg) {
                memPtr = (OclMemory *)(&tmpOclImg);
            }
            memPtr->resize(desc);
            U32 w = desc.dims[0];
            U32 pr = UNI_ALIGN(w, alignVal) - w;
            memPtr->padding(0, pr, 0, 0);
            matrixTran->desc = memPtr->get_desc();
            if (tmpImg) {
                matrixTran->mem = tmpImg->mem;
            } else {
                CHECK_STATUS(gcl_create_sub_buffer(
                    matrixTran->desc.byteSize, tmpOff, tmpbuf, &(matrixTran->mem)));
            }
            CHECK_STATUS(ocl_data_trans_form(handle, &curMatrix, matrixTran, 0, 0, NCHWC4_TO_NCHW));
            *descTran = desc;
            return SUCCESS;
        }
        memPtr->resize(desc);
        tmpMatrix.desc = memPtr->get_desc();
        CHECK_STATUS(
            gcl_create_sub_buffer(tmpMatrix.desc.byteSize, tmpOff, tmpbuf, &(tmpMatrix.mem)));
        CHECK_STATUS(ocl_data_trans_form(handle, &curMatrix, &tmpMatrix, 0, 0, NCHWC4_TO_NCHW));
        curMatrix = tmpMatrix;
    }
    if (needReshape) {
        if (curTrans == targetTrans) {
            U32 w, h, c;
            get_desc_whc(desc, &c, &h, &w);
            desc.dims[0] = h;
            desc.dims[1] = w;
            tmpOclMem.resize(desc);
            tmpMatrix.desc = tmpOclMem.get_desc();
            CHECK_STATUS(
                gcl_create_sub_buffer(tmpMatrix.desc.byteSize, tmpOff, tmpbuf, &(tmpMatrix.mem)));
            CHECK_STATUS(trans_matmul_input(handle, &curMatrix, &tmpMatrix, dt, w, h, c));
            curTrans = !curTrans;
            curMatrix = tmpMatrix;
        } else if (curMatrix.desc.num != tensorNumElements(desc)) {
            tmpOclMem.resize(desc);
            tmpMatrix.desc = tmpOclMem.get_desc();
            CHECK_STATUS(
                gcl_create_sub_buffer(tmpMatrix.desc.byteSize, tmpOff, tmpbuf, &(tmpMatrix.mem)));
            CHECK_STATUS(ocl_data_trans_form(handle, &curMatrix, &tmpMatrix, 0, 0, NCHW_TO_NCHW));
            curMatrix = tmpMatrix;
        }
        tmpOclMem.resize(descReshaped);
        curMatrix.desc = tmpOclMem.get_desc();
        desc = descReshaped;
    }

    if (curTrans == targetTrans) {
        if (tmpImg) {
            tmpOclImg.resize(desc);
            matrixTran->desc = tmpOclImg.get_desc();
            matrixTran->mem = tmpImg->mem;
            CHECK_STATUS(ocl_data_trans_form(handle, &curMatrix, matrixTran, 0, 0, NCHW_TO_NCHW));
        } else {
            matrixTran->desc = curMatrix.desc;
            matrixTran->mem = curMatrix.mem;
        }
    } else {
        U32 w, h, c;
        get_desc_whc(desc, &c, &h, &w);
        desc.dims[0] = h;
        desc.dims[1] = w;
        memPtr = (tmpImg) ? (OclMemory *)(&tmpOclImg) : &tmpOclMem;
        memPtr->resize(desc);
        U32 pr = UNI_ALIGN(h, alignVal) - h;
        memPtr->padding(0, pr, 0, 0);
        matrixTran->desc = memPtr->get_desc();
        if (tmpImg) {
            matrixTran->mem = tmpImg->mem;
        } else {
            CHECK_STATUS(gcl_create_sub_buffer(
                matrixTran->desc.byteSize, tmpOff, tmpbuf, &(matrixTran->mem)));
        }
        CHECK_STATUS(trans_matmul_input(handle, &curMatrix, matrixTran, dt, w, h, c));
    }
    *descTran = desc;
    return SUCCESS;
}

inline EE process_matmul_output(TensorDesc matrixADesc,
    TensorDesc matrixBDesc,
    TensorDesc matrixCDesc,
    U32 *tmpOff,
    GCLMem_t tmpbuf,
    GCLMem_t matrixC,
    TensorDesc *matrixCDescTran,
    GCLMem_t matrixCTran)
{
    U32 ac, ah, aw;
    U32 bc, bh, bw;
    get_desc_whc(matrixADesc, &ac, &ah, &aw);
    get_desc_whc(matrixBDesc, &bc, &bh, &bw);
    if (ah != bh || ac != bc) {
        CHECK_STATUS(NOT_MATCH);
    }
    TensorDesc desc;
    desc.dt = matrixCDesc.dt;
    desc.nDims = 3;
    desc.df = DF_MTK;
    desc.dims[0] = bw;
    desc.dims[1] = aw;
    desc.dims[2] = ac;
    *matrixCDescTran = desc;

    OclMemory tmpOclMem;
    tmpOclMem.resize(desc);
    matrixCTran->desc = tmpOclMem.get_desc();
    if (tensorNumElements(matrixCDesc) == matrixC->desc.num) {
        matrixCTran->mem = matrixC->mem;
    } else {
        CHECK_STATUS(
            gcl_create_sub_buffer(matrixCTran->desc.byteSize, tmpOff, tmpbuf, &(matrixCTran->mem)));
    }
    return SUCCESS;
}

inline EE matmul_tn_core_mali_fp16(GCLHandle_t handle,
    TensorDesc matrixADesc,
    GCLMem_t matrixA,
    TensorDesc matrixBDesc,
    GCLMem_t matrixB,
    TensorDesc biasDesc,
    GCLMem_t bias,
    GCLMem_t tmpBuf,
    TensorDesc matrixCDesc,
    GCLMem_t matrixC,
    ForwardRunInfoMali_t forwardRunInfo)
{
    DataType dt = matrixADesc.dt;
    U32 ac, ah, aw;
    U32 bc, bh, bw;
    U32 cc, ch, cw;
    U32 aw_str, ah_str, aw_off, ah_off;
    U32 bw_str, bh_str, bw_off, bh_off;
    U32 cw_str, ch_str, cw_off, ch_off;
    get_desc_whc(matrixADesc, &ac, &ah, &aw);
    get_desc_whc(matrixBDesc, &bc, &bh, &bw);
    get_desc_whc(matrixCDesc, &cc, &ch, &cw);
    CHECK_STATUS(gclmem_get_desc_padding(matrixA->desc, &aw_str, &ah_str, NULL, &aw_off, &ah_off));
    CHECK_STATUS(gclmem_get_desc_padding(matrixB->desc, &bw_str, &bh_str, NULL, &bw_off, &bh_off));
    CHECK_STATUS(gclmem_get_desc_padding(matrixC->desc, &cw_str, &ch_str, NULL, &cw_off, &ch_off));
    if (ah != bh) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (ac != bc || ac != cc) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (aw != ch || bw != cw) {
        CHECK_STATUS(NOT_MATCH);
    }

    U32 item_a = forwardRunInfo->best_k[0];
    U32 item_b = forwardRunInfo->best_h[0];
    U32 M = aw_str;
    U32 N = bw_str;
    U32 K = ah;
    U32 A_str = aw_str * ah_str;
    U32 B_str = bw_str * bh_str;
    U32 C_str = cw_str * ch_str;
    U32 A_off = ah_off * aw_str + aw_off;
    U32 B_off = bh_off * bw_str + bw_off;
    U32 C_off = ch_off * cw_str + cw_off;
    Mem A = matrixA->mem;
    Mem B = matrixB->mem;
    Mem C = matrixC->mem;
    Mem biasMem = tmpBuf->mem;
    OclGemmBiasMode biasMode = NO_BIAS;
    if (bias) {
        if (biasDesc.dims[0] == aw) {
            biasMode = USE_BIAS_MATCH_A;
        } else if (biasDesc.dims[1] == bw) {
            biasMode = USE_BIAS_MATCH_B;
        } else {
            CHECK_STATUS(NOT_MATCH);
        }
        biasMem = bias->mem;
    }
    GCLMemType matrixAMemType = matrixA->desc.memType;
    GCLMemType matrixBMemType = matrixB->desc.memType;
    GCLMemType matrixCMemType = matrixC->desc.memType;
    char kernelName[128];
    Kernel kernel;
    KernelOpt kernelOpt;
    U32 gs[3] = {(cw + item_b - 1) / item_b, (ch + item_a - 1) / item_a, cc};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;

    CHECK_STATUS(set_gemm_tn_opt_mali(item_a, item_b, biasMode, false, {}, dt,
        matrixAMemType, matrixBMemType, matrixCMemType, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, M, N, K, A_str, B_str, C_str, A_off, B_off, C_off,
        cw_str, cw, ch, cc, gs[0], gs[1], A, B, biasMem, C));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
    return SUCCESS;
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
    ForwardRunInfoMali_t forwardRunInfo)
{
    TensorDesc matrixADescReshaped;
    TensorDesc matrixBDescReshaped;
    bool needReshapeA;
    bool needReshapeB;
    GCLMemType amt = gclmemMatrixADesc.memType;
    GCLMemType bmt = gclmemMatrixBDesc.memType;
    get_reshaped_desc(matrixADesc, matrixBDesc, transposeA, transposeB, &needReshapeA,
        &needReshapeB, &matrixADescReshaped, &matrixBDescReshaped);
    bool needProcessA = need_process_matmul_input(matrixADesc, amt, needReshapeA, transposeA, true);
    bool needProcessB = need_process_matmul_input(matrixBDesc, bmt, needReshapeB, transposeB, false);
    U32 bufSize = 0;
    U32 imgASize[3] = {0};
    U32 imgBSize[3] = {0};
    if (needProcessA) {
        U32 item_a = forwardRunInfo->best_k[0];
        get_process_matmul_input_tmp_bytes(matrixADesc, matrixADescReshaped, gclmemMatrixADesc,
            needReshapeA, transposeA, true, item_a, &bufSize, imgASize);
    }
    if (needProcessB) {
        U32 item_b = forwardRunInfo->best_h[0];
        get_process_matmul_input_tmp_bytes(matrixBDesc, matrixBDescReshaped, gclmemMatrixBDesc,
            needReshapeB, transposeB, false, item_b, &bufSize, imgBSize);
    }
    if (needReshapeA || needReshapeB) {
        if (tensorNumElements(matrixCDesc) != gclmemMatrixCDesc.num) {
            bufSize += UNI_ALIGN(tensorNumBytes(matrixCDesc), BUFFER_ALIGN_BASE);
        }
    }
    bytes[0] = bufSize;
    bytes[1] = imgASize[0];
    bytes[2] = imgASize[1];
    bytes[3] = imgASize[2];
    bytes[4] = imgBSize[0];
    bytes[5] = imgBSize[1];
    bytes[6] = imgBSize[2];
    return SUCCESS;
}

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
    ForwardRunInfoMali_t forwardRunInfo)
{
    CHECK_STATUS(matmul_checkpara_mali_fp16(matrixADesc, matrixBDesc, matrixCDesc));
    CHECK_STATUS(fill_output_zero(handle, matrixC, matrixCDesc));
    TensorDesc matrixADescReshaped;
    TensorDesc matrixBDescReshaped;
    bool needReshapeA;
    bool needReshapeB;
    get_reshaped_desc(matrixADesc, matrixBDesc, transposeA, transposeB, &needReshapeA,
        &needReshapeB, &matrixADescReshaped, &matrixBDescReshaped);
    GCLMemType amt = matrixA->desc.memType;
    GCLMemType bmt = matrixB->desc.memType;
    bool needProcessA = need_process_matmul_input(matrixADesc, amt, needReshapeA, transposeA, true);
    bool needProcessB = need_process_matmul_input(matrixBDesc, bmt, needReshapeB, transposeB, false);
    GCLMem_t matrixAPtr = matrixA;
    GCLMem_t matrixBPtr = matrixB;
    GCLMem_t matrixCPtr = matrixC;
    U32 tmpOff = 0;
    GCLMem matrixTran[3];
    TensorDesc descTran;
    if (needProcessA) {
        U32 item_a = forwardRunInfo->best_k[0];
        CHECK_STATUS(
            process_matmul_input(handle, matrixADesc, matrixADescReshaped, matrixA, needReshapeA,
                transposeA, true, item_a, &tmpOff, tmp[0], tmp[1], &descTran, &(matrixTran[0])));
        matrixADesc = descTran;
        matrixAPtr = &(matrixTran[0]);
    }
    if (needProcessB) {
        U32 item_b = forwardRunInfo->best_h[0];
        CHECK_STATUS(
            process_matmul_input(handle, matrixBDesc, matrixBDescReshaped, matrixB, needReshapeB,
                transposeB, false, item_b, &tmpOff, tmp[0], tmp[2], &descTran, &(matrixTran[1])));
        matrixBDesc = descTran;
        matrixBPtr = &(matrixTran[1]);
    }
    if (needReshapeA || needReshapeB) {
        CHECK_STATUS(process_matmul_output(matrixADesc, matrixBDesc, matrixCDesc, &tmpOff, tmp[0],
            matrixC, &descTran, &(matrixTran[2])));
        matrixCDesc = descTran;
        matrixCPtr = &(matrixTran[2]);
    }
    CHECK_STATUS(matmul_tn_core_mali_fp16(handle, matrixADesc, matrixAPtr, matrixBDesc, matrixBPtr,
        biasDesc, bias, tmp[0], matrixCDesc, matrixCPtr, forwardRunInfo));
    if (matrixCPtr->mem != matrixC->mem) {
        CHECK_STATUS(ocl_data_trans_form(handle, matrixCPtr, matrixC, 0, 0, NCHW_TO_NCHW));
    }
    return SUCCESS;
}
