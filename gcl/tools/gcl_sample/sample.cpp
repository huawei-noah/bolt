// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



#include"gcl.h"
#include"libkernelbin.h"

void setMemDesc(GCLMem_t mem, DataType dt, DataFormat ft, GCLMemType mt,
                U32 s0, U32 s1, U32 s2, U32 off0, U32 off1, U32 off2){
    mem->desc.stride[0] = s0;
    mem->desc.stride[1] = s1;
    mem->desc.stride[2] = s2;
    mem->desc.offset[0] = off0;
    mem->desc.offset[1] = off1;
    mem->desc.offset[2] = off2;
    mem->desc.num = s0 * s1 * s2;
    mem->desc.byteSize = s0 * s1 * s2 * bytesOf(dt);
    mem->desc.memFormat = ft;
    mem->desc.memType = mt;
}

int main(I32 argc , I8* argv[]){
    GCLHandle_t handle;
    CHECK_STATUS(gcl_create_handle(&handle));
    U32 iw, ih, ic, in;
    U32 fw, fh, fc, fn;
    U32 sv, pv;
    U32 ow, oh, oc, on;

    iw = 4;
    ih = 4;
    ic = 3;
    in = 1;

    fw = 1;
    fh = 1;
    fc = 3;
    fn = 1;

    ow = iw;
    oh = ih;
    oc = fn;
    on = in;

    sv = 1;
    pv = 0;

    GCLMem input, flt, bias, output;
    setMemDesc(&input, DT_F16, DF_NCHW, GCL_MEM_BUF, iw * ic, ih, 1, pv, pv, 0);
    setMemDesc(&flt,   DT_F16, DF_NCHW, GCL_MEM_BUF, fw, fh, fc + fn, 0,  0,  0);
    setMemDesc(&bias,  DT_F16, DF_NCHW, GCL_MEM_BUF, fn, 1,  1,  0,  0,  0);
    setMemDesc(&output,DT_F16, DF_NCHW, GCL_MEM_BUF, ow, oh, oc, 0,  0,  0);

    CHECK_STATUS(gcl_create_memory(handle, &input));
    CHECK_STATUS(gcl_create_memory(handle, &flt));
    CHECK_STATUS(gcl_create_memory(handle, &bias));
    CHECK_STATUS(gcl_create_memory(handle, &output));

    U8* iptr = new U8[input.desc.byteSize];
    U8* fptr = new U8[flt.desc.byteSize];

    F16* ival = (F16*)iptr;
    F16* fval = (F16*)fptr;
    for(int i = 0; i < input.desc.num; i++){
        ival[i] = i % 12;
    }

    for(int i = 0; i < flt.desc.num; i++){
        fval[i] = i + 1;
    }

    CHECK_STATUS(gcl_trans_memory(handle, (void*)iptr, (void*)&input, &input.desc.byteSize, HOST_TO_DEVICE_BUF, CL_TRUE));
    CHECK_STATUS(gcl_trans_memory(handle, (void*)fptr, (void*)&flt,   &flt.desc.byteSize,   HOST_TO_DEVICE_BUF, CL_TRUE));
    CHECK_STATUS(gcl_print_memory<F16>(handle, &input, "input"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, &flt,   "flt"));


    Kernel kernel;
    char kernelname[128];
    U32 item_w, item_h, ew;
/*
    item_w = 4;
    for(int i = 0; i < 4; i++){
        ew = ow % item_w;
        item_h = 1 << i;
        sprintf(kernelname, "conv_direct_s%d_spe_ck_%d%d%d%d%d%d", sv, fw, fc, fn, item_w, item_h, ew);
        CHECK_STATUS(gcl_create_kernel_binary(handle, kernelname, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw, iw * ih, ow, pv, pv, ow, oh, input.mem, flt.mem, bias.mem, output.mem));
        U32 gs[3];
        gs[0] = (ow + item_w - 1) / item_w;
        gs[1] = (oh + item_h - 1) / item_h;
        U32 dim   = 2;
        U32 ls[3] = {16, 16, 1};
        std::cout << gs[0] << " " << gs[1] << std::endl;
        CHECK_STATUS(gcl_run_kernel_profiling(handle, kernel, dim, gs, ls, kernelname));
    }

    item_w = 8;
    for(int i = 0; i < 4; i++){
        ew = ow % item_w;
        item_h = 1 << i;
        sprintf(kernelname, "conv_direct_s%d_spe_ck_%d%d%d%d%d%d", sv, fw, fc, fn, item_w, item_h, ew);
        CHECK_STATUS(gcl_create_kernel_binary(handle, kernelname, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw, iw * ih, ow, pv, pv, ow, oh, input.mem, flt.mem, bias.mem, output.mem));
        U32 gs[3];
        gs[0] = (ow + item_w - 1) / item_w;
        gs[1] = (oh + item_h - 1) / item_h;
        U32 dim   = 2;
        U32 ls[3] = {16, 16, 1};
        std::cout << gs[0] << " " << gs[1] << std::endl;
        CHECK_STATUS(gcl_run_kernel_profiling(handle, kernel, dim, gs, ls, kernelname));
    }

    item_w = 1;
    for(int i = 0; i < 4; i++){
        ew = ow % item_w;
        item_h = 1 << i;
        sprintf(kernelname, "conv_direct_s%d_spe_ck_%d%d%d%d%d%d", sv, fw, fc, fn, item_w, item_h, ew);
        CHECK_STATUS(gcl_create_kernel_binary(handle, kernelname, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw, iw * ih, ow, pv, pv, ow, oh, input.mem, flt.mem, bias.mem, output.mem));
        U32 gs[3];
        gs[0] = (ow + item_w - 1) / item_w;
        gs[1] = (oh + item_h - 1) / item_h;
        U32 dim   = 2;
        U32 ls[3] = {16, 16, 1};
        std::cout << gs[0] << " " << gs[1] << std::endl;
        CHECK_STATUS(gcl_run_kernel_profiling(handle, kernel, dim, gs, ls, kernelname));
    }
*/
    for(int i = 0; i < 1; i++){
        sprintf(kernelname, "mytest");
        CHECK_STATUS(gcl_create_kernel_binary(handle, kernelname, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw * ic, iw * ih, ow, pv, pv, ow, oh, input.mem, flt.mem, bias.mem, output.mem));
        U32 gs[3];
        gs[0] = ow / 2;
        gs[1] = oh;
        U32 dim   = 2;
        U32 ls[3] = {16, 16, 1};
        CHECK_STATUS(gcl_run_kernel_profiling(handle, kernel, dim, gs, ls, kernelname));
    }

    CHECK_STATUS(gcl_print_memory<F16>(handle, &output, "output"));
    delete[] iptr;
    delete[] fptr;
    CHECK_STATUS(gcl_release_memory(&input));
    CHECK_STATUS(gcl_release_memory(&flt));
    CHECK_STATUS(gcl_release_memory(&bias));
    CHECK_STATUS(gcl_release_memory(&output));
    CHECK_STATUS(release_kernel(kernel));
    gcl_destroy_handle(handle);
}



