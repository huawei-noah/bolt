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
#include "type.h"
#include "tensor_desc.h"
#include "error.h"
#include "tensor_computing_type.h"
#include "gpu/mali/uchar/bilateral_slice_apply_mali_uchar.h"

inline EE bilateral_slice_apply_checkpara_mali_uchar(TensorDesc inputDesc,
                                                     TensorDesc guideDesc,
                                                     TensorDesc gridDesc,
                                                     TensorDesc outputDesc) {
    if (inputDesc.dt != outputDesc.dt || inputDesc.dt != DT_U8)  return NOT_SUPPORTED;
    if (gridDesc.dt  != guideDesc.dt  || gridDesc.dt  != DT_F16) return NOT_SUPPORTED;
    return SUCCESS; 
}

inline EE bilateral_slice_apply_core_mali_uchar(GCLHandle_t             handle,
                                               TensorDesc              inputDesc,
                                               const GCLMem_t          input,
                                               TensorDesc              guideDesc,
                                               const GCLMem_t          guide,
                                               TensorDesc              gridDesc,
                                               const GCLMem_t          grid,
                                               BilateralSliceApplyDesc bilateralSliceApplyDesc,
                                               ForwardRunInfoMali_t    forwardRunInfo,
                                               GCLMem_t                tmpBuf,
                                               TensorDesc              outputDesc,
                                               GCLMem_t                output){
    UNUSED(guideDesc);                                               
    UNUSED(forwardRunInfo);                                               
    U32 iw, ih, ic, in;
    U32 gw, gh, gc, gn;
    U32 ow, oh, oc, on;
    tensorSelectGet(inputDesc,  NULL, NULL, &in, &ic, &ih, &iw);
    tensorSelectGet(gridDesc,   NULL, NULL, &gn, &gc, &gh, &gw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);

    U32  coe = bilateralSliceApplyDesc.coefficient_len;
    BilateralSliceApplyMode mode = bilateralSliceApplyDesc.mode;
    U32  dep = gc / coe;
    U32  gcw = gc * gw;
    U32   wh = iw * ih;
    F32 scale_x = (F32)gw / iw;
    F32 scale_y = (F32)gh / ih;
    Mem inbuf, gridbuf, guidebuf, outbuf, gridTran;
    inbuf    = input->mem;
    gridbuf  = grid->mem;
    outbuf   = output->mem;
    gridTran = tmpBuf->mem;
    if(mode == BSliceApply_NULL) {
        guidebuf = guide->mem;
    } else {
        guidebuf = inbuf;
    }

    Kernel kernel;
    CHECK_STATUS(gcl_create_kernel_binary(handle, "bilateral_slice_apply_pre", &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, gh, gc, gcw, scale_y, gridbuf, gridTran));
    U32 gs0[3] = {gc / 4, gw, ih};
    U32 ls0[3] = {16, 16, 1};
    U32 dim0   = 3;
    gcl_set_kernelVec(handle, kernel, dim0, gs0, ls0);
    
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel_profiling(handle, kernel, dim0, gs0, ls0, "bilateral_slice_apply_pre"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, grid,  "bilateral_slice_apply_grid"));
#endif
    char kernelname[128];
    if(mode == BSliceApply_CONV) {
        sprintf(kernelname, "bilateral_slice_apply_c12_conv_uchar");
    } else {
        sprintf(kernelname, "bilateral_slice_apply_c12_uchar");
    }
    CHECK_STATUS(gcl_create_kernel_binary(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw, wh, gc, gw, gh, gcw, dep, coe, scale_x, scale_y, guidebuf, gridTran, inbuf, outbuf));
    U32 gs[2] = {ow, oh};
    U32 ls[2] = {16, 16};
    U32 dim   = 2;
    gcl_set_kernelVec(handle, kernel, dim, gs, ls);

#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel_profiling(handle, kernel, dim, gs, ls, kernelname));
    CHECK_STATUS(gcl_print_memory<F16>(handle, input,  "bilateral_slice_apply_input"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, output, "bilateral_slice_apply_output"));
    if(mode == BSliceApply_NULL)CHECK_STATUS(gcl_print_memory<F16>(handle, guide,  "bilateral_slice_apply_guide"));
#endif
    return SUCCESS; 
     
}

EE bilateral_slice_apply_mali_uchar(GCLHandle_t             handle,
                                   TensorDesc              inputDesc,
                                   const GCLMem_t          input,
                                   TensorDesc              guideDesc,
                                   const GCLMem_t          guide,
                                   TensorDesc              gridDesc,
                                   const GCLMem_t          grid,
                                   BilateralSliceApplyDesc bilateralSliceApplyDesc,
                                   ForwardRunInfoMali_t    forwardRunInfo,
                                   U32                     tmpBytes,
                                   GCLMem_t                tmpBuf,
                                   TensorDesc              outputDesc,
                                   GCLMem_t                output){
    UNUSED(tmpBytes);
    CHECK_STATUS(bilateral_slice_apply_checkpara_mali_uchar(inputDesc, guideDesc, gridDesc, outputDesc)); 
    CHECK_STATUS(bilateral_slice_apply_core_mali_uchar(handle, inputDesc, input, guideDesc, guide, gridDesc, grid, bilateralSliceApplyDesc, forwardRunInfo, tmpBuf, outputDesc, output));
    return SUCCESS; 
}

