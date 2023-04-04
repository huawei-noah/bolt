// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/rnn_mali_fp16.h"
#include "gpu/mali/fp16/gemv_mali_fp16.h"
#include "gpu/mali/fp16/matmul_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/gemm_tn_opt.h"
#include "gpu/mali/cl/kernel_option/copy_opt.h"
#include "gpu/mali/cl/kernel_option/transpose_opt.h"
#include "gpu/mali/cl/kernel_option/rnncell_update_res_opt.h"

inline EE reshape_input(
    GCLHandle_t handle, U32 item, GCLMem_t input, GCLMem_t tmpBuf, GCLMem_t sMem, U32 *subMemOff)
{
    MemTransFormType type = (input->desc.memFormat == DF_NCHW) ? NCHW_TO_NCHW : NCHWC4_TO_NCHW;
    DataType dt = input->desc.dt;
    U32 dim[3] = {1, 1, 1};
    for (U32 i = 0; i < input->desc.nDims; i++) {
        if (i < 2) {
            dim[i] = input->desc.dims[i];
        } else {
            dim[2] = dim[2] * input->desc.dims[i];
        }
    }
    GCLMem tMem;
    tMem.desc = input->desc;
    U32 str[3] = {dim[0], dim[1], dim[2]};
    U32 off[3] = {0, 0, 0};
    MemFlags flag = CL_MEM_READ_WRITE;
    CHECK_STATUS(
        gclmem_set_desc_padding(&(tMem.desc), str, off, dt, DF_NCHW, GCL_MEM_BUF, flag));
    CHECK_STATUS(gcl_create_sub_buffer(tMem.desc.byteSize, subMemOff, tmpBuf, &(tMem.mem)));
    CHECK_STATUS(ocl_data_trans_form(handle, input, &tMem, 0, 0, type));
    *sMem = tMem;
    return SUCCESS;
}

inline EE gemm_trans_input(
    GCLHandle_t handle, U32 item, TensorDesc inputDesc, GCLMem_t sMem, GCLMem_t dMem)
{
    TensorDesc desc = inputDesc;
    desc.nDims = 2;
    desc.dims[1] = inputDesc.dims[inputDesc.nDims - 1];
    desc.dims[1] *= inputDesc.dims[inputDesc.nDims - 2];
    desc.dims[0] = inputDesc.dims[inputDesc.nDims - 3];
    for (U32 i = 0; i < inputDesc.nDims - 3; i++) {
        desc.dims[0] *= inputDesc.dims[i];
    }
    OclMemory tmpMem;
    OclMemoryImg tmpImg;
    OclMemory *memPtr = &tmpMem;
    memPtr->resize(desc);
    sMem->desc = memPtr->get_desc();

    U32 w = desc.dims[0];
    U32 h = desc.dims[1];
    desc.dims[0] = h;
    desc.dims[1] = w;
    U32 pr = UNI_ALIGN(h, item) - h;
    if (dMem->desc.memType != GCL_MEM_BUF) {
        memPtr = (OclMemory *)&(tmpImg);
    }
    memPtr->resize(desc);
    memPtr->padding(0, pr, 0, 0);
    dMem->desc = memPtr->get_desc();
    CHECK_STATUS(trans_matmul_input(handle, sMem, dMem, desc.dt, w, h, 1));
    return SUCCESS;
}

inline EE rnn_core_matmul(GCLHandle_t handle,
    DataType dt,
    U32 M,
    U32 N,
    U32 K,
    U32 item_n,
    U32 item_m,
    GCLMemType at,
    GCLMemType bt,
    Mem gemmMatA,
    Mem gemmMatB,
    Mem gemmBias,
    Mem gemmMatC)
{
    U32 N64 = UNI_ALIGN(N, 64);
    U32 A_str = M * K;
    U32 B_str = N * K;
    U32 C_str = M * N64;

    U32 gs[3] = {N / item_n, M / item_m, 1};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    char kernelName[128];
    Kernel kernel;
    KernelOpt kernelOpt;
    CHECK_STATUS(set_gemm_tn_opt_mali(item_m, item_n, USE_BIAS_MATCH_B, false, {},
        dt, at, bt, GCL_MEM_BUF, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, M, N, K, A_str, B_str, C_str, 0, 0, 0, N64, N, M, 1,
        gs[0], gs[1], gemmMatA, gemmMatB, gemmBias, gemmMatC));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    //CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    return SUCCESS;
}

inline EE rnn_core_init_state(GCLHandle_t handle,
    DataType dt,
    U32 inputNum,
    U32 col,
    U32 hDim,
    U32 item_c,
    bool backDir,
    GCLMem_t input,
    Mem stateC,
    Mem stateH)
{
    GCLMem stateCMem;
    GCLMem stateHMem;
    stateCMem.mem = stateC;
    stateHMem.mem = stateH;
    GCLMemDesc desc;
    U32 str[3] = {
        UNI_ALIGN(col, 4),
        1,
        1,
    };
    U32 off[3] = {0, 0, 0};
    MemFlags flag = CL_MEM_READ_WRITE;
    CHECK_STATUS(gclmem_set_desc_padding(&desc, str, off, dt, DF_NCHW, GCL_MEM_BUF, flag));
    stateCMem.desc = desc;
    CHECK_STATUS(ocl_fill_memory_zero(handle, &stateCMem, 0));

    str[0] = UNI_ALIGN(hDim, item_c);
    CHECK_STATUS(gclmem_set_desc_padding(&desc, str, off, dt, DF_NCHW, GCL_MEM_BUF, flag));
    stateHMem.desc = desc;
    CHECK_STATUS(ocl_fill_memory_zero(handle, &stateHMem, 0));

    if (inputNum > 1) {
        char kernelName[128];
        KernelOpt kernelOpt;
        Kernel kernel;
        if (input[1].desc.memType != GCL_MEM_BUF) {
            CHECK_STATUS(NOT_SUPPORTED)
        }
        CHECK_STATUS(set_copy_opt_mali(false, dt, kernelName, &kernelOpt));
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        U32 gs = (col + 3) / 4;
        U32 ls = 0;
        U32 dim = 1;
        CHECK_STATUS(gcl_set_kernelArgs(kernel, col, col, 0, 0, gs, input[1].mem, stateC));
        gcl_set_kernelVec(handle, kernel, dim, &gs, &ls, kernelName);
#ifdef _DEBUG
//        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, &gs, &ls, kernelName));
#endif

        U32 hMemOff = col;
        cl_mem hSrcMem = input[1].mem;
        if (inputNum > 2) {
            if (input[2].desc.memType != GCL_MEM_BUF) {
                CHECK_STATUS(NOT_SUPPORTED)
            }
            hMemOff = 0;
            hSrcMem = input[2].mem;
        }
        if (backDir) {
            hMemOff += hDim;
        }
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        gs = (hDim + 3) / 4;
        CHECK_STATUS(gcl_set_kernelArgs(kernel, hDim, hDim, hMemOff, 0, gs, hSrcMem, stateH));
        gcl_set_kernelVec(handle, kernel, dim, &gs, &ls, kernelName);
#ifdef _DEBUG
//        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, &gs, &ls, kernelName));
#endif
    }
    return SUCCESS;
}

inline EE rnn_core_build_gemv_kernel_info(GCLHandle_t handle,
    DataType dt,
    U32 item_c,
    U32 row,
    U32 *tmpOff,
    GCLMem_t tmpBuf,
    Mem *reduceMem,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    return gemv_build_run_info(handle, item_c, row, 1, {}, true, false, dt, tmpOff,
        tmpBuf, reduceMem, kernelName, kernelOpt);
}

inline EE rnn_core_gemv(GCLHandle_t handle,
    U32 gemvRow,
    U32 gemvCol,
    U32 item_c,
    Mem vec,
    Mem mat,
    Mem bias,
    Mem out,
    Mem tmp,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    return gemv_run(handle, item_c, gemvRow, gemvCol, 1, 0, 0, 0, 0, vec, mat, bias, tmp, out,
        kernelName, kernelOpt);
}

inline EE rnn_core_build_update_kernel_info(DataType dt, bool useProject, char *kernelName, KernelOpt *kernelOpt)
{
    CHECK_STATUS(set_rnncell_update_res_opt_mali(
        useProject, true, dt, GCL_MEM_BUF, GCL_MEM_BUF, kernelName, kernelOpt));
    return SUCCESS;
}

inline EE rnn_core_update(GCLHandle_t handle,
    U32 col,
    U32 out_off,
    RNNParamSpec rnnPara,
    Mem stateC,
    Mem stateH,
    Mem interMem,
    Mem out,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    float fbias = rnnPara.forget_bias;
    float zonecell = rnnPara.zoneout_cell;
    float zoneout = rnnPara.zoneout_output;
    U32 gs = (col + 3) / 4;
    U32 ls = 16;
    U32 dim = 1;
    Kernel kernel;

    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(
        kernel, col, out_off, gs, fbias, zonecell, zoneout, stateC, stateH, interMem, out));
    gcl_set_kernelVec(handle, kernel, dim, &gs, &ls, kernelName);
#ifdef _DEBUG
//    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, &gs, &ls, kernelName));
#endif
    return SUCCESS;
}

inline EE rnn_core_copy_stateC(GCLHandle_t handle, DataType dt, U32 col, Mem stateC, GCLMem_t output)
{
    char kernelName[128];
    Kernel kernel;
    KernelOpt kernelOpt;
    CHECK_STATUS(set_copy_opt_mali(false, dt, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    U32 gs = (col + 3) / 4;
    U32 ls = 0;
    U32 dim = 1;
    CHECK_STATUS(gcl_set_kernelArgs(kernel, col, col, 0, 0, gs, stateC, output[1].mem));
    gcl_set_kernelVec(handle, kernel, dim, &gs, &ls, kernelName);
#ifdef _DEBUG
//    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, &gs, &ls, kernelName));
#endif
    return SUCCESS;
}

inline EE rnn_core_copy_stateH(
    GCLHandle_t handle, DataType dt, U32 col, U32 hDim, U32 outputNum, bool backDir, Mem stateH, GCLMem_t output)
{
    char kernelName[128];
    Kernel kernel;
    KernelOpt kernelOpt;
    CHECK_STATUS(set_copy_opt_mali(false, dt, kernelName, &kernelOpt));
    U32 hMemOff = col;
    cl_mem hDstMem = output[1].mem;
    if (outputNum > 2) {
        hMemOff = 0;
        hDstMem = output[2].mem;
    }
    if (backDir) {
        hMemOff += hDim;
    }
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    U32 gs = (hDim + 3) / 4;
    U32 ls = 0;
    U32 dim = 1;
    CHECK_STATUS(gcl_set_kernelArgs(kernel, hDim, hDim, hMemOff, 0, gs, stateH, hDstMem));
    gcl_set_kernelVec(handle, kernel, dim, &gs, &ls, kernelName);
#ifdef _DEBUG
//    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, &gs, &ls, kernelName));
#endif
    return SUCCESS;
}

inline EE rnn_core_mali_fp16(GCLHandle_t handle,
    std::vector<TensorDesc> inputDescs,
    GCLMem_t input,
    std::vector<TensorDesc> filterDescs,
    GCLMem_t filter,
    std::vector<TensorDesc> biasDescs,
    GCLMem_t bias,
    RNNParamSpec rnnPara,
    GCLMem_t tmpBuf,
    GCLMem_t tmpImg,
    std::vector<TensorDesc> outputDescs,
    GCLMem_t output,
    ForwardRunInfoMali_t forwardRunInfo)
{
    bool project = (rnnPara.num_projection > 0) ? true : false;
    if (project) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    cl_mem gemmMatA, gemmMatB, gemmMatC, gemmBias;
    U32 filterCount = 0;
    U32 biasCount = 0;
    gemmMatB = filter[filterCount].mem;
    GCLMemType gemmMatBType = filter[filterCount].desc.memType;
    gemmBias = bias[biasCount].mem;
    filterCount++;
    biasCount++;

    U32 item_n = forwardRunInfo->best_h[0];
    U32 item_m = forwardRunInfo->best_k[0];
    GCLMem sMem = input[0];
    GCLMem dMem;
    U32 subMemOff = 0;
    if (needReshapeInput(input[0].desc)) {
        CHECK_STATUS(reshape_input(handle, item_m, &input[0], tmpBuf, &sMem, &subMemOff));
    }

    TensorDesc desc = inputDescs[0];
    DataType dt = desc.dt;
    U32 batch = desc.dims[desc.nDims - 1];
    U32 step = desc.dims[desc.nDims - 2];
    U32 xDim = desc.dims[desc.nDims - 3];
    U32 hDim = rnnPara.num_outputs;
    U32 col = (rnnPara.num_projection > 0) ? rnnPara.num_projection : hDim;
    for (U32 i = 0; i < desc.nDims - 3; i++) {
        xDim *= desc.dims[i];
    }

    GCLMemType gemmMatAType = GCL_MEM_BUF;
    if (tmpImg) {
        gemmMatA = tmpImg->mem;
        gemmMatAType = GCL_MEM_IMG_3D;
    } else {
        U32 gemmMatASize = UNI_ALIGN(batch * step, item_m) * xDim * bytesOf(desc.dt);
        CHECK_STATUS(gcl_create_sub_buffer(gemmMatASize, &subMemOff, tmpBuf, &gemmMatA));
    }
    dMem.mem = gemmMatA;
    dMem.desc.memType = gemmMatAType;
    CHECK_STATUS(gemm_trans_input(handle, item_m, desc, &sMem, &dMem));

    U32 M = UNI_ALIGN(step * batch, item_m);
    U32 K = xDim;
    U32 N = UNI_ALIGN(4 * col, item_n);
    U32 N64 = UNI_ALIGN(N, 64);
    U32 gemmMatCSize = M * N64 * bytesOf(desc.dt);
    U32 gemvBiasBase = subMemOff;
    CHECK_STATUS(gcl_create_sub_buffer(gemmMatCSize, &subMemOff, tmpBuf, &gemmMatC));
    CHECK_STATUS(rnn_core_matmul(handle, dt, M, N, K, item_n, item_m, gemmMatAType, gemmMatBType,
        gemmMatA, gemmMatB, gemmBias, gemmMatC));

    cl_mem stateC, stateH;
    U32 item_c = forwardRunInfo->best_c[1];
    U32 item_k = forwardRunInfo->best_k[1];
    U32 c_align = (item_c > 16) ? (item_c >> 4) : item_c;
    U32 stateCSize = UNI_ALIGN(col, 4) * bytesOf(desc.dt);
    U32 stateHSize = UNI_ALIGN(hDim, c_align) * bytesOf(desc.dt);
    CHECK_STATUS(gcl_create_sub_buffer(stateCSize, &subMemOff, tmpBuf, &stateC));
    CHECK_STATUS(gcl_create_sub_buffer(stateHSize, &subMemOff, tmpBuf, &stateH));
    CHECK_STATUS(rnn_core_init_state(
        handle, dt, inputDescs.size(), col, hDim, c_align, false, input, stateC, stateH));

    std::vector<Mem> gemvBiasVec;
    for (U32 i = 0; i < step; i++) {
        Mem gemvBias = NULL;
        U32 gemvBiasSize = N64 * bytesOf(desc.dt);
        CHECK_STATUS(gcl_create_sub_buffer(gemvBiasSize, &gemvBiasBase, tmpBuf, &gemvBias));
        gemvBiasVec.push_back(gemvBias);
    }

    U32 filterRow = 4 * col;
    Mem interMem, gemvFilterMem;
    U32 interMemSize = filterRow * bytesOf(desc.dt);
    CHECK_STATUS(gcl_create_sub_buffer(interMemSize, &subMemOff, tmpBuf, &interMem));
    gemvFilterMem = filter[filterCount].mem;
    filterCount++;

    char kernelNameGemv[128];
    KernelOpt kernelOptGemv;
    U32 tmpOff = subMemOff;
    Mem reduceMem = tmpBuf->mem;
    CHECK_STATUS(rnn_core_build_gemv_kernel_info(
        handle, dt, item_c, filterRow, &tmpOff, tmpBuf, &reduceMem, kernelNameGemv, &kernelOptGemv));

    char kernelNameUpdate[128];
    KernelOpt kernelOptUpdate;
    CHECK_STATUS(rnn_core_build_update_kernel_info(dt, project, kernelNameUpdate, &kernelOptUpdate));

    U32 ow_str, oh_str, ow_off, oh_off;
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off));
    U32 out_off = ow_off + ow_str * oh_off;
    cl_mem outbuf = output->mem;

    for (U32 i = 0; i < step; i++) {
        CHECK_STATUS(rnn_core_gemv(handle, filterRow, hDim, item_c, stateH, gemvFilterMem,
            gemvBiasVec[i], interMem, reduceMem, kernelNameGemv, &kernelOptGemv));
        CHECK_STATUS(rnn_core_update(handle, col, out_off, rnnPara, stateC, stateH, interMem,
            outbuf, kernelNameUpdate, &kernelOptUpdate));
        out_off += ow_str;
    }

    if (outputDescs.size() > 1) {
        CHECK_STATUS(rnn_core_copy_stateC(handle, dt, col, stateC, output));
        CHECK_STATUS(
            rnn_core_copy_stateH(handle, dt, col, hDim, outputDescs.size(), false, stateH, output));
    }

    if (rnnPara.bi_direction) {
        gemmMatB = filter[filterCount].mem;
        gemmMatBType = filter[filterCount].desc.memType;
        gemmBias = bias[biasCount].mem;
        filterCount++;
        biasCount++;
        CHECK_STATUS(rnn_core_matmul(handle, dt, M, N, K, item_n, item_m, gemmMatAType, gemmMatBType,
            gemmMatA, gemmMatB, gemmBias, gemmMatC));
        CHECK_STATUS(rnn_core_init_state(
            handle, dt, inputDescs.size(), col, hDim, c_align, true, input, stateC, stateH));
        gemvFilterMem = filter[filterCount].mem;
        filterCount++;
        out_off -= hDim;
        for (I32 i = step - 1; i >= 0; i--) {
            CHECK_STATUS(rnn_core_gemv(handle, filterRow, hDim, item_c, stateH, gemvFilterMem,
                gemvBiasVec[i], interMem, reduceMem, kernelNameGemv, &kernelOptGemv));
            CHECK_STATUS(rnn_core_update(handle, col, out_off, rnnPara, stateC, stateH, interMem,
                outbuf, kernelNameUpdate, &kernelOptUpdate));
            out_off -= ow_str;
        }
        if (outputDescs.size() > 1) {
            CHECK_STATUS(
                rnn_core_copy_stateH(handle, dt, col, hDim, outputDescs.size(), true, stateH, output));
        }
    }
    return SUCCESS;
}

inline void transform_filter_desc(TensorDesc filterDesc,
    RNNParamSpec rnnPara,
    U32 item_n,
    U32 item_c,
    U32 item_k,
    TensorDesc *ftmDesc)
{
    DataType fdt;
    U32 filterRow, filterCol;
    tensorSelectGet(filterDesc, &fdt, NULL, NULL, NULL, &filterRow, &filterCol);
    U32 hDim = rnnPara.num_outputs;
    U32 xDim = filterCol - hDim;

    TensorDesc desc;
    desc.df = DF_NCHW;
    desc.dt = fdt;
    desc.nDims = 4;
    desc.dims[3] = 1;
    desc.dims[2] = 1;
    desc.dims[1] = xDim;
    desc.dims[0] = UNI_ALIGN(filterRow, item_n);
    ftmDesc[0] = desc;

    filterDesc.dims[0] = hDim;
    ftmDesc[1] = gemv_transform_filter_desc(filterDesc, 0, item_c, item_k);
}

EE rnn_transform_filter_bytes_mali_fp16(TensorDesc filterDesc,
    RNNParamSpec rnnPara,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc *ftmDesc)
{
    U32 item_n = forwardRunInfo->best_h[0];
    U32 item_c = forwardRunInfo->best_c[1];
    U32 item_k = forwardRunInfo->best_k[1];
    transform_filter_desc(filterDesc, rnnPara, item_n, item_c, item_k, ftmDesc);
    return SUCCESS;
}

EE rnn_transform_filter_mali_fp16(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    GCLMem_t tmpBuf,
    RNNParamSpec rnnPara,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem,
    ForwardRunInfoMali_t forwardRunInfo)
{
    DataType fdt;
    U32 filterRow, filterCol;
    tensorSelectGet(filterDesc, &fdt, NULL, NULL, NULL, &filterRow, &filterCol);
    U32 hDim = rnnPara.num_outputs;
    U32 xDim = filterCol - hDim;
    char kernelName[128];
    KernelOpt kernelOpt;
    Kernel kernel;
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim = 2;
    cl_mem weightGemm;
    cl_mem weightGemv;
    U32 subMemOff = 0;
    U32 weightGemmSize = xDim * filterRow * bytesOf(fdt);
    U32 weightGemvSize = hDim * filterRow * bytesOf(fdt);
    CHECK_STATUS(gcl_create_sub_buffer(weightGemmSize, &subMemOff, tmpBuf, &weightGemm));
    CHECK_STATUS(gcl_create_sub_buffer(weightGemvSize, &subMemOff, tmpBuf, &weightGemv));

    U32 biDirNum = (rnnPara.bi_direction) ? 2 : 1;
    U32 filterCount = 0;
    U32 filterTranCount = 0;
    U32 item_n = forwardRunInfo->best_h[0];
    U32 item_c = forwardRunInfo->best_c[1];
    U32 item_k = forwardRunInfo->best_k[1];
    TensorDesc gemmDesc = filterDesc;
    gemmDesc.dims[0] = xDim;
    TensorDesc gemvDesc = filterDesc;
    gemvDesc.dims[0] = hDim;

    OclMemory tmpMem;
    GCLMem gemmWei;
    tmpMem.resize(gemmDesc);
    gemmWei.mem = weightGemm;
    gemmWei.desc = tmpMem.get_desc();
    for (U32 i = 0; i < biDirNum; i++) {
        gs[0] = (xDim + 3) / 4 + (hDim + 3) / 4;
        gs[1] = filterRow;
        Mem filterMem = filter[filterCount].mem;
        filterCount++;

        CHECK_STATUS(set_common_opt(
            fdt, GCL_MEM_BUF, GCL_MEM_BUF, "rnn_split_weight", kernelName, &kernelOpt));
        CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelName, &kernel, &kernelOpt));
        CHECK_STATUS(
            gcl_set_kernelArgs(kernel, xDim, hDim, gs[0], gs[1], filterMem, weightGemm, weightGemv));
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));

        CHECK_STATUS(trans_matmul_input(
            handle, &gemmWei, fltmem + filterTranCount, gemmDesc.dt, xDim, filterRow, 1, false));
        filterTranCount++;

        Mem filterTranMem = fltmem[filterTranCount].mem;
        filterTranCount++;
        CHECK_STATUS(gemv_transform_filter_run(
            handle, hDim, filterRow, item_c, fdt, weightGemv, filterTranMem));
    }
    transform_filter_desc(filterDesc, rnnPara, item_n, item_c, item_k, fltmemDesc);
    return SUCCESS;
}

EE rnn_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    GCLMemDesc gclmemInputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    RNNParamSpec rnnPara,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo)
{
    U32 size = 0;
    U32 item_n = forwardRunInfo->best_h[0];
    U32 item_m = forwardRunInfo->best_k[0];

    if (needReshapeInput(gclmemInputDesc)) {
        size = tensorNumBytes(inputDesc);
        size = UNI_ALIGN(size, BUFFER_ALIGN_BASE);
    }
    U32 batch = inputDesc.dims[inputDesc.nDims - 1];
    U32 step = inputDesc.dims[inputDesc.nDims - 2];
    U32 xDim = inputDesc.dims[inputDesc.nDims - 3];
    for (U32 i = 0; i < inputDesc.nDims - 3; i++) {
        xDim *= inputDesc.dims[i];
    }
    bool useImg = check_qualcomm_device();
    if (useImg) {
        U32 w = (batch * step + 3) / 4;
        U32 h = xDim;
        if (CHECK_MEET_IMAGE_LIMITS(w, h, 1)) {
            bytes[1] = w;
            bytes[2] = h;
            bytes[3] = 1;
        } else {
            useImg = false;
        }
    }
    if (!useImg) {
        U32 gemmMatASize = UNI_ALIGN(batch * step, item_m) * xDim * bytesOf(inputDesc.dt);
        size += UNI_ALIGN(gemmMatASize, BUFFER_ALIGN_BASE);
    }

    U32 hDim = rnnPara.num_outputs;
    U32 col = (rnnPara.num_projection > 0) ? rnnPara.num_projection : hDim;
    U32 filterRow = col * 4;
    U32 M = UNI_ALIGN(step * batch, item_m);
    U32 N = UNI_ALIGN(filterRow, item_n);
    U32 N64 = UNI_ALIGN(N, 64);
    U32 gemmMatCSize = M * N64 * bytesOf(inputDesc.dt);
    size += UNI_ALIGN(gemmMatCSize, BUFFER_ALIGN_BASE);

    U32 item_c = forwardRunInfo->best_c[1];
    U32 item_k = forwardRunInfo->best_k[1];
    U32 c_align = (item_c > 16) ? (item_c >> 4) : item_c;
    U32 stateCSize = UNI_ALIGN(col, 4) * bytesOf(inputDesc.dt);
    U32 stateHSize = UNI_ALIGN(hDim, c_align) * bytesOf(inputDesc.dt);
    size += UNI_ALIGN(stateCSize, BUFFER_ALIGN_BASE);
    size += UNI_ALIGN(stateHSize, BUFFER_ALIGN_BASE);

    U32 interMemSize = filterRow * bytesOf(inputDesc.dt);
    size += UNI_ALIGN(interMemSize, BUFFER_ALIGN_BASE);

    if (item_c > 16) {
        U32 reduceMemSize = filterRow * 32 * bytesOf(inputDesc.dt);
        size += UNI_ALIGN(reduceMemSize, BUFFER_ALIGN_BASE);
    }

    U32 fltTranTmpSize = xDim * filterRow * bytesOf(inputDesc.dt);
    fltTranTmpSize =
        UNI_ALIGN(fltTranTmpSize, BUFFER_ALIGN_BASE) + hDim * filterRow * bytesOf(inputDesc.dt);
    bytes[0] = (fltTranTmpSize > size) ? fltTranTmpSize : size;
    return SUCCESS;
}

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
    ForwardRunInfoMali_t forwardRunInfo)
{
    for (U32 i = 0; i < outputDescs.size(); i++) {
        CHECK_STATUS(fill_output_zero(handle, &output[i], outputDescs[i]));
    }
    CHECK_STATUS(rnn_core_mali_fp16(handle, inputDescs, input, filterDescs, filter, biasDescs, bias,
        rnnPara, tmp[0], tmp[1], outputDescs, output, forwardRunInfo));
    return SUCCESS;
}
