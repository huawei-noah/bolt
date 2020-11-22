// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#ifdef _USE_FP16
#include "gcl.h"
#include "types.h"
#include "libkernelsource.h"

void setMemDesc(GCLMem_t mem,
    DataType dt,
    DataFormat ft,
    GCLMemType mt,
    U32 s0,
    U32 s1,
    U32 s2,
    U32 off0,
    U32 off1,
    U32 off2)
{
    mem->desc.stride[0] = s0 + 2 * off0;
    mem->desc.stride[1] = s1 + 2 * off1;
    mem->desc.stride[2] = s2;
    mem->desc.offset[0] = off0;
    mem->desc.offset[1] = off1;
    mem->desc.offset[2] = off2;
    mem->desc.num = (s0 + 2 * off0) * (s1 + 2 * off0) * s2;
    mem->desc.byteSize = s0 * s1 * s2 * bytesOf(dt);
    mem->desc.memFormat = ft;
    mem->desc.memType = mt;
}

int main()
{
    GCLHandle_t handle;
    CHECK_STATUS(gcl_create_handle(&handle));
    U32 iw, ih, ic, in;
    U32 fw, fh, fc, fn;
    U32 sv, pv;
    U32 ow, oh, oc, on;

    iw = 1440;
    ih = 960;
    ic = 4;
    in = 1;

    fw = 3;
    fh = 3;
    fc = 4;
    fn = 1;

    ow = iw;
    oh = ih;
    oc = fn;
    on = in;

    sv = 1;
    pv = 1;

    TensorDesc outDesc = tensor4d(DT_F16, on, oc, oh, ow);
    GCLMem_t input = gcl_create_gclmem();
    GCLMem_t flt = gcl_create_gclmem();
    GCLMem_t bias = gcl_create_gclmem();
    GCLMem_t output = gcl_create_gclmem();
    setMemDesc(input, DT_F16, DF_NCWHC4, GCL_MEM_BUF, iw + 8, ih, ic, pv, pv, 0);
    setMemDesc(flt, DT_F16, DF_NCWHC4, GCL_MEM_BUF, fw * fh, fc, fn, 0, 0, 0);
    setMemDesc(bias, DT_F16, DF_NCHW, GCL_MEM_BUF, fn, 1, 1, 0, 0, 0);
    setMemDesc(output, DT_F16, DF_NCHW, GCL_MEM_BUF, ow, oh, oc * 4, 0, 0, 0);
    CHECK_STATUS(gcl_create_memory(handle, input));
    CHECK_STATUS(gcl_create_memory(handle, flt));
    CHECK_STATUS(gcl_create_memory(handle, bias));
    CHECK_STATUS(gcl_create_memory(handle, output));

    U8 *iptr = new U8[input->desc.byteSize];
    U8 *fptr = new U8[flt->desc.byteSize];
    U8 *bptr = new U8[bias->desc.byteSize];

    F16 *ival = (F16 *)iptr;
    F16 *fval = (F16 *)fptr;
    F16 *bval = (F16 *)bptr;
    for (U32 i = 0; i < input->desc.num; i++) {
        ival[i] = (rand() & 1023) / 1024.0 - 0.5;
        U32 s0 = input->desc.stride[0];
        U32 s1 = input->desc.stride[1];
        U32 j = i % (s0 * s1);
        U32 h = j % s1;
        U32 w = j / s1;
        if (h < pv || w < pv) {
            ival[i] = 0;
        }
        if (h >= ih + pv || w >= iw + pv) {
            ival[i] = 0;
        }
    }

    for (U32 i = 0; i < flt->desc.num; i++) {
        fval[i] = (rand() & 1023) / 1024.0 - 0.5;
    }

    for (U32 i = 0; i < bias->desc.num; i++) {
        bval[i] = (rand() & 1023) / 1024.0 - 0.5;
    }

    CHECK_STATUS(gcl_trans_memory(
        handle, (void *)iptr, (void *)input, &input->desc.byteSize, HOST_TO_DEVICE_BUF, CL_TRUE));
    CHECK_STATUS(gcl_trans_memory(
        handle, (void *)fptr, (void *)flt, &flt->desc.byteSize, HOST_TO_DEVICE_BUF, CL_TRUE));
    CHECK_STATUS(gcl_trans_memory(
        handle, (void *)bptr, (void *)bias, &bias->desc.byteSize, HOST_TO_DEVICE_BUF, CL_TRUE));

    Kernel kernel;
    char kernelname[128];
    U32 be = 0;
    U32 end = 0;
    for (int i = 3; i <= 8; i++) {
        sprintf(kernelname, "conv_direct_s1_fn_spe_nchw_3%d", i);
        CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
        U32 iw_str = input->desc.stride[0];
        U32 ih_str = input->desc.stride[1];
        U32 ihw_str = iw_str * ih_str;
        U32 ic_str = (input->desc.stride[2] + 3) / 4;
        U32 ih_off = input->desc.offset[0] - pv;
        U32 iw_off = input->desc.offset[1] - pv;
        U32 sh = 1;
        U32 ow_str = output->desc.stride[0];
        U32 oh_str = output->desc.stride[1];
        U32 ohw_str = ow_str * oh_str;
        U32 ow_off = output->desc.offset[0];
        U32 oh_off = output->desc.offset[1];

        U32 gs[2];
        gs[0] = oh;
        gs[1] = (ow + i - 1) / i;
        U32 dim = 2;
        U32 ls[2] = {0, 0};
        CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, ihw_str, ic_str, ih_off, iw_off, oh_str,
            ow_str, ohw_str, oh_off, ow_off, ow, sh, gs[0], gs[1], input->mem, flt->mem, bias->mem,
            output->mem));
        CHECK_STATUS(gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname));
        end = handle->kernelVec.size();
        CHECK_STATUS(gcl_run_kernelVec_timing(handle, be, end));
        CHECK_STATUS(gcl_run_kernelVec_timing(handle, be, end));
        CHECK_STATUS(gcl_run_kernelVec_timing(handle, be, end));
        CHECK_STATUS(gcl_run_kernelVec_timing(handle, be, end));
        be = end;
#ifdef _DEBUG
        CHECK_STATUS(gcl_check_data<F16>(handle, outDesc, output, 0, false));
#endif
    }

    delete[] iptr;
    delete[] fptr;
    delete[] bptr;
    gcl_destroy_gclmem(input);
    gcl_destroy_gclmem(flt);
    gcl_destroy_gclmem(bias);
    gcl_destroy_gclmem(output);
    gcl_destroy_handle(handle);
}
#endif
