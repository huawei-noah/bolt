// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/generate_proposals_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/common_opt.h"
#include <queue>

inline EE generate_proposals_checkpara_mali_fp16(TensorDesc deltaDesc,
    TensorDesc logitDesc,
    TensorDesc imgInfoDesc,
    TensorDesc anchorDesc,
    TensorDesc outputDesc)
{
    if (deltaDesc.dt != logitDesc.dt || deltaDesc.dt != imgInfoDesc.dt ||
        deltaDesc.dt != anchorDesc.dt || deltaDesc.dt != outputDesc.dt) {
        return NOT_MATCH;
    }
    return SUCCESS;
}

inline EE process_delta(GCLHandle_t handle,
    TensorDesc deltaDesc,
    GCLMem_t delta,
    U32 *tmpBufOff,
    GCLMem_t tmpBuf,
    U32 *dw_str,
    U32 *dh_str,
    U32 *d_off,
    GCLMemType *dmemType,
    Mem *dmem)
{
    DataType dt;
    U32 dw, dh, dc, dn;
    tensorSelectGet(deltaDesc, &dt, NULL, &dn, &dc, &dh, &dw);
    GCLMem tmem;
    tmem.desc = delta->desc;
    tmem.desc.df = DF_NCHWC4;
    U32 str[3] = {dw, dh, (dc + 3) / 4 * dn};
    U32 off[3] = {0, 0, 0};
    MemFlags flag = CL_MEM_READ_WRITE;
    CHECK_STATUS(gclmem_set_desc_padding(&(tmem.desc), str, off, dt, DF_NCHWC4, GCL_MEM_BUF, flag));
    U32 size = tmem.desc.byteSize;
    CHECK_STATUS(gcl_create_sub_buffer(size, tmpBufOff, tmpBuf, &(tmem.mem)));
    CHECK_STATUS(ocl_data_trans_form(handle, delta, &tmem, 0, 0, NCHW_TO_NCHWC4));
    *dw_str = dw;
    *dh_str = dh;
    *d_off = 0;
    *dmem = tmem.mem;
    *dmemType = GCL_MEM_BUF;
    return SUCCESS;
}

inline GCLMemDesc build_proposals_gpu_desc(TensorDesc deltaDesc, GCLMemDesc deltaMemDesc)
{
    DataType dt;
    U32 dw, dh, dc, dn;
    tensorSelectGet(deltaDesc, &dt, NULL, &dn, &dc, &dh, &dw);
    GCLMemDesc desc = deltaMemDesc;
    desc.df = DF_NCHWC4;
    U32 str[3] = {dw, dh, (dc + 3) / 4 * dn};
    U32 off[3] = {0, 0, 0};
    MemFlags flag = CL_MEM_READ_WRITE;
    CHECK_STATUS(gclmem_set_desc_padding(&desc, str, off, dt, DF_NCHWC4, GCL_MEM_BUF, flag));
    return desc;
}

inline EE generate_proposals_apply_delta(GCLHandle_t handle,
    TensorDesc deltaDesc,
    GCLMem_t delta,
    TensorDesc imgInfoDesc,
    GCLMem_t imgInfo,
    TensorDesc anchorDesc,
    GCLMem_t anchor,
    U32 *tmpBufOff,
    GCLMem_t tmpBuf,
    GCLMem_t proposalGpu)
{
    DataType dt;
    DataFormat ddf;
    U32 dw, dh, dc, dn;
    tensorSelectGet(deltaDesc, &dt, &ddf, &dn, &dc, &dh, &dw);
    U32 dw_str, dh_str, dw_off, dh_off, d_off;
    CHECK_STATUS(gclmem_get_desc_padding(delta->desc, &dw_str, &dh_str, NULL, &dw_off, &dh_off));
    d_off = dh_off * dw_str + dw_off;
    GCLMemType dmemType = delta->desc.memType;
    Mem dmem = delta->mem;
    if (ddf != DF_NCHWC4) {
        CHECK_STATUS(process_delta(handle, deltaDesc, delta, tmpBufOff, tmpBuf, &dw_str, &dh_str,
            &d_off, &dmemType, &dmem));
    }

    proposalGpu->desc = build_proposals_gpu_desc(deltaDesc, delta->desc);
    U32 psize = proposalGpu->desc.byteSize;
    CHECK_STATUS(gcl_create_sub_buffer(psize, tmpBufOff, tmpBuf, &(proposalGpu->mem)));
    U32 pw_str, ph_str, pw_off, ph_off, p_off;
    CHECK_STATUS(
        gclmem_get_desc_padding(proposalGpu->desc, &pw_str, &ph_str, NULL, &pw_off, &ph_off));
    p_off = ph_off * pw_str + pw_off;
    GCLMemType pmemType = proposalGpu->desc.memType;
    Mem pmem = proposalGpu->mem;

    Kernel kernel;
    char kernelName[128];
    KernelOpt kernelOpt;
    CHECK_STATUS(set_common_opt(
        dt, dmemType, pmemType, "generate_proposals_apply_deltas", kernelName, &kernelOpt));
    U32 gs[3] = {dw, dh, (dc + 3) / 4};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, dw_str, dh_str, pw_str, ph_str, d_off, p_off, dw, dh,
        gs[0], gs[1], anchor->mem, imgInfo->mem, dmem, pmem));
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    return SUCCESS;
}

inline EE process_logit(GCLHandle_t handle,
    TensorDesc logitDesc,
    GCLMem_t logit,
    U32 *tmpBufOff,
    GCLMem_t tmpBuf,
    GCLMem_t logitGpu)
{
    if (logitDesc.df != DF_NCHWC4 && logit->desc.num == tensorNumElements(logitDesc)) {
        return SUCCESS;
    }
    logitGpu->desc.df = DF_NCHW;
    DataType dt;
    DataFormat df;
    U32 w, h, c, n;
    tensorSelectGet(logitDesc, &dt, &df, &n, &c, &h, &w);
    U32 str[3] = {w, h, c * n};
    U32 off[3] = {0, 0, 0};
    MemFlags flag = CL_MEM_READ_WRITE;
    CHECK_STATUS(
        gclmem_set_desc_padding(&(logitGpu->desc), str, off, dt, DF_NCHW, GCL_MEM_BUF, flag));
    U32 size = logitGpu->desc.byteSize;
    CHECK_STATUS(gcl_create_sub_buffer(size, tmpBufOff, tmpBuf, &(logitGpu->mem)));
    MemTransFormType type = (df == DF_NCHWC4) ? NCHWC4_TO_NCHW : NCHW_TO_NCHW;
    CHECK_STATUS(ocl_data_trans_form(handle, logit, logitGpu, 0, 0, type));
    return SUCCESS;
}

inline EE update_output(GCLHandle_t handle,
    I32 validIndexNum,
    I32 proposalLen,
    GCLMem_t postTopIndex,
    GCLMem_t proposal,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    U32 oh = outputDesc.dims[1];  //postNmsTop
    U32 ow_str, oh_str, ow_off, oh_off, o_off;
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off));
    o_off = oh_off * ow_str + ow_off;
    if (validIndexNum < 0) {
        CHECK_STATUS(NOT_MATCH);
    }

    Kernel kernel;
    char kernelName[128];
    KernelOpt kernelOpt;
    CHECK_STATUS(set_common_opt(outputDesc.dt, GCL_MEM_BUF, GCL_MEM_BUF,
        "generate_proposals_update_output", kernelName, &kernelOpt));
    U32 gs[1] = {oh};
    U32 ls[1] = {0};
    U32 dim = 1;
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, ow_str, o_off, validIndexNum, proposalLen, gs[0],
        postTopIndex->mem, proposal->mem, output->mem))
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    return SUCCESS;
}

template <typename T>
inline EE update_output_v2(
    GCLHandle_t handle, I32 validIndexNum, T *outputCpu, TensorDesc outputDesc, GCLMem_t output)
{
    U32 ow_str, oh_str, ow_off, oh_off, o_off;
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off));
    U32 ow = outputDesc.dims[0];
    if (ow_str != ow) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (validIndexNum < 0) {
        CHECK_STATUS(NOT_MATCH);
    }
    U32 oh = outputDesc.dims[1];                //postNmsTop
    for (U32 i = validIndexNum; i < oh; i++) {  //fill zero when validIndexNum < postNmsTop
        outputCpu[i * 4] = 0;
        outputCpu[i * 4 + 1] = 0;
        outputCpu[i * 4 + 2] = 0;
        outputCpu[i * 4 + 3] = 0;
    }
    o_off = oh_off * ow_str + ow_off;
    U32 size = tensorNumBytes(outputDesc);
    U32 offSize = o_off * bytesOf(outputDesc.dt);
    CHECK_STATUS(
        gcl_trans_memory(handle, outputCpu, output, &size, HOST_TO_DEVICE_BUF, CL_TRUE, &offSize));
    return SUCCESS;
}

template <typename T>
inline void sort_topk(I32 k, I32 len, T *val, I32 *sortIndex)
{
    auto cmp = [val](int a, int b) { return val[a] < val[b]; };
    std::priority_queue<int, std::vector<int>, decltype(cmp)> queue(cmp);
    for (I32 i = 0; i < len; i++) {
        queue.push(i);
    }
    for (I32 i = 0; i < k; i++) {
        sortIndex[i] = queue.top();
        queue.pop();
    }
}

template <typename T>
F32 iou(T *a, T *b)
{
    F32 area1 = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
    F32 area2 = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
    F32 x11 = (a[0] > b[0]) ? a[0] : b[0];
    F32 y11 = (a[1] > b[1]) ? a[1] : b[1];
    F32 x22 = (a[2] < b[2]) ? a[2] : b[2];
    F32 y22 = (a[3] < b[3]) ? a[3] : b[3];
    F32 w = x22 - x11 + 1;
    F32 h = y22 - y11 + 1;
    if (w < 0)
        w = 0;
    if (h < 0)
        h = 0;
    F32 I32er = w * h;
    return I32er / (area1 + area2 - I32er);
}

template <typename T>
inline int nms(I32 preTopK,
    I32 postTopK,
    F32 thresh,
    I32 *sortIndex,
    I32 *curIndex,
    const T *proposals,
    I32 *postTopIndex)
{
    int *indexA = sortIndex;
    int *indexB = curIndex;
    int vaildNum = postTopK;
    T a[4];
    T b[4];
    int indexANum = preTopK;
    int indexBNum = 0;
    for (I32 i = 0; i < postTopK; i++) {
        U32 ia = indexA[0];
        a[0] = proposals[ia * 4];
        a[1] = proposals[ia * 4 + 1];
        a[2] = proposals[ia * 4 + 2];
        a[3] = proposals[ia * 4 + 3];
        postTopIndex[i] = ia;
        for (I32 j = 1; j < indexANum; j++) {
            U32 ib = indexA[j];
            b[0] = proposals[ib * 4];
            b[1] = proposals[ib * 4 + 1];
            b[2] = proposals[ib * 4 + 2];
            b[3] = proposals[ib * 4 + 3];
            if (iou<T>(a, b) <= thresh) {
                indexB[indexBNum] = ib;
                indexBNum++;
            }
        }
        int *indexTmp = indexA;
        indexA = indexB;
        indexB = indexTmp;
        indexANum = indexBNum;
        indexBNum = 0;
        if (indexANum == 0) {
            vaildNum = i + 1;
            break;
        }
    }
    return vaildNum;
}

template <typename T>
inline int nms_v2(I32 preTopK,
    I32 postTopK,
    F32 thresh,
    I32 *sortIndex,
    I32 *curIndex,
    const T *proposals,
    T *postTopProposals)
{
    int *indexA = sortIndex;
    int *indexB = curIndex;
    int vaildNum = postTopK;
    T a[4];
    T b[4];
    int indexANum = preTopK;
    int indexBNum = 0;
    for (I32 i = 0; i < postTopK; i++) {
        U32 ia = indexA[0];
        a[0] = proposals[ia * 4];
        a[1] = proposals[ia * 4 + 1];
        a[2] = proposals[ia * 4 + 2];
        a[3] = proposals[ia * 4 + 3];
        postTopProposals[i * 4] = a[0];
        postTopProposals[i * 4 + 1] = a[1];
        postTopProposals[i * 4 + 2] = a[2];
        postTopProposals[i * 4 + 3] = a[3];
        for (I32 j = 1; j < indexANum; j++) {
            U32 ib = indexA[j];
            b[0] = proposals[ib * 4];
            b[1] = proposals[ib * 4 + 1];
            b[2] = proposals[ib * 4 + 2];
            b[3] = proposals[ib * 4 + 3];
            if (iou<T>(a, b) <= thresh) {
                indexB[indexBNum] = ib;
                indexBNum++;
            }
        }
        int *indexTmp = indexA;
        indexA = indexB;
        indexB = indexTmp;
        indexANum = indexBNum;
        indexBNum = 0;
        if (indexANum == 0) {
            vaildNum = i + 1;
            break;
        }
    }
    return vaildNum;
}

inline EE generate_proposals_core_mali_fp16(GCLHandle_t handle,
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
    U32 tmpBufOff = 0;
    GCLMem proposalGpu;
    CHECK_STATUS(generate_proposals_apply_delta(handle, deltaDesc, delta, imgInfoDesc, imgInfo,
        anchorDesc, anchor, &tmpBufOff, tmpBuf, &proposalGpu));
    GCLMem logitGpu = *logit;
    CHECK_STATUS(process_logit(handle, logitDesc, logit, &tmpBufOff, tmpBuf, &logitGpu));

    U32 proposalLen = tensorNumElements(logitDesc);
    U32 preNmsTop = generateProposalsParam.pre_nms_topN;
    U32 postNmsTop = generateProposalsParam.post_nms_topN;
    float thresh = generateProposalsParam.nms_thresh;

    U32 cpuBytesOff = 0;
    F16 *proposalCpu = (F16 *)tmpCpu;
    U32 proposalCpuSize = tensorNumBytes(deltaDesc);
    CHECK_STATUS(gcl_trans_memory(
        handle, &proposalGpu, proposalCpu, &proposalCpuSize, DEVICE_BUF_TO_HOST, CL_TRUE));
    cpuBytesOff += proposalCpuSize;

    F16 *logitCpu = (F16 *)(tmpCpu + cpuBytesOff);
    U32 logitCpuSize = tensorNumBytes(logitDesc);
    CHECK_STATUS(
        gcl_trans_memory(handle, &logitGpu, logitCpu, &logitCpuSize, DEVICE_BUF_TO_HOST, CL_TRUE));
    cpuBytesOff += logitCpuSize;

    U32 sortIndexSize = preNmsTop * bytesOf(DT_I32);
    I32 *sortIndex = (I32 *)(tmpCpu + cpuBytesOff);
    sort_topk<F16>(preNmsTop, proposalLen, logitCpu, sortIndex);
    cpuBytesOff += sortIndexSize;

    U32 curIndexSize = sortIndexSize;
    I32 *curIndex = (I32 *)(tmpCpu + cpuBytesOff);
    cpuBytesOff += sortIndexSize;

    F16 *outputCpu = (F16 *)(tmpCpu + cpuBytesOff);
    I32 validIndexNum =
        nms_v2<F16>(preNmsTop, postNmsTop, thresh, sortIndex, curIndex, proposalCpu, outputCpu);
    CHECK_STATUS(update_output_v2<F16>(handle, validIndexNum, outputCpu, outputDesc, output));

    //    I32* postTopIndex = (I32*)(tmpCpu + cpuBytesOff);
    //    I32 validIndexNum = nms<F16>(preNmsTop, postNmsTop, thresh, sortIndex, curIndex, proposalCpu, postTopIndex);
    //    U32 postTopIndexSize = postNmsTop * bytesOf(DT_I32);
    //    GCLMem postTopIndexGpu;
    //    CHECK_STATUS(gcl_create_sub_buffer(postTopIndexSize, &tmpBufOff, tmpBuf, &(postTopIndexGpu.mem)));
    //    CHECK_STATUS(gcl_trans_memory(handle, postTopIndex, &postTopIndexGpu, &postTopIndexSize, HOST_TO_DEVICE_BUF, CL_TRUE));
    //    CHECK_STATUS(update_output(handle, validIndexNum, proposalLen, &postTopIndexGpu, &proposalGpu, outputDesc, output));
    return SUCCESS;
}

EE generate_proposals_infer_forward_tmp_bytes_mali_fp16(TensorDesc deltaDesc,
    TensorDesc logitDesc,
    GCLMemDesc gclMemLogitDesc,
    GenerateProposalsParamSpec generateProposalsParam,
    U32 *bytes)
{
    U32 gpuSize = 0;
    U32 cpuSize = 0;
    U32 preNmsTop = generateProposalsParam.pre_nms_topN;
    U32 postNmsTop = generateProposalsParam.post_nms_topN;

    DataType dt;
    U32 dw, dh, dc, dn;
    tensorSelectGet(deltaDesc, &dt, NULL, &dn, &dc, &dh, &dw);
    U32 size = dw * dh * ((dc + 3) / 4 * 4) * dn * bytesOf(dt);
    gpuSize += UNI_ALIGN(size, BUFFER_ALIGN_BASE);
    if (deltaDesc.df != DF_NCHWC4) {
        gpuSize *= 2;
    }
    U32 proposalLen = tensorNumElements(logitDesc);
    if (logitDesc.df == DF_NCHWC4 || gclMemLogitDesc.num != proposalLen) {
        gpuSize += UNI_ALIGN(tensorNumBytes(logitDesc), BUFFER_ALIGN_BASE);
    }
    //gpuSize += UNI_ALIGN(postNmsTop * bytesOf(DT_I32), BUFFER_ALIGN_BASE);

    cpuSize += size;
    cpuSize += tensorNumBytes(logitDesc);
    cpuSize += preNmsTop * bytesOf(DT_I32) * 2;
    //cpuSize += postNmsTop * bytesOf(DT_I32);
    cpuSize += postNmsTop * 4 * bytesOf(DT_F16);
    bytes[0] = gpuSize;
    bytes[1] = cpuSize;
    return SUCCESS;
}
EE generate_proposals_mali_fp16(GCLHandle_t handle,
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
    CHECK_STATUS(generate_proposals_checkpara_mali_fp16(
        deltaDesc, logitDesc, imgInfoDesc, anchorDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(generate_proposals_core_mali_fp16(handle, deltaDesc, delta, logitDesc, logit,
        imgInfoDesc, imgInfo, anchorDesc, anchor, generateProposalsParam, tmpBuf, tmpCpu,
        outputDesc, output));
    return SUCCESS;
}
