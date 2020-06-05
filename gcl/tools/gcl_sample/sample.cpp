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
    mem->desc.stride[0] = s0 + 2 * off0;
    mem->desc.stride[1] = s1 + 2 * off1;
    mem->desc.stride[2] = s2;
    mem->desc.offset[0] = off0;
    mem->desc.offset[1] = off1;
    mem->desc.offset[2] = off2;
    mem->desc.num = s0 * s1 * s2;
    mem->desc.byteSize = s0 * s1 * s2 * bytesOf(dt);
    mem->desc.memFormat = ft;
    mem->desc.memType = mt;
}

int main(){
while(1) {
    GCLHandle_t handle;
    CHECK_STATUS(gcl_create_handle(&handle));
    CHECK_STATUS(gcl_regist_binMap(handle));
    U32 iw, ih, ic, in;
    U32 fw, fh, fc, fn;
    U32 sv, pv;
    U32 ow, oh, oc, on;

    iw = 4;
    ih = 4;
    ic = 4;
    in = 1;

    fw = 3;
    fh = 3;
    fc = 4;
    fn = 4;

    ow = iw;
    oh = ih;
    oc = fn;
    on = in;

    sv = 1;
    pv = 1;

    GCLMem_t input  = gcl_create_gclmem();
    GCLMem_t flt    = gcl_create_gclmem();
    GCLMem_t bias   = gcl_create_gclmem();
    GCLMem_t output = gcl_create_gclmem();
    setMemDesc(input, DT_F16, DF_NCHW, GCL_MEM_BUF, iw, ih,  ic, pv, pv, 0);
    setMemDesc(flt,   DT_F16, DF_NCHW, GCL_MEM_BUF, fw * fh, fc, fn, 0,  0,  0);
    setMemDesc(bias,  DT_F16, DF_NCHW, GCL_MEM_BUF, fn,      1,  1,  0,  0,  0);
    setMemDesc(output,DT_F16, DF_NCHW, GCL_MEM_BUF, ow,      oh, oc, 0,  0,  0);
    CHECK_STATUS(gcl_create_memory(handle, input));
    CHECK_STATUS(gcl_create_memory(handle, flt));
    CHECK_STATUS(gcl_create_memory(handle, bias));
    CHECK_STATUS(gcl_create_memory(handle, output));

    U8* iptr = new U8[input->desc.byteSize];
    U8* fptr = new U8[flt->desc.byteSize];
    U8* bptr = new U8[bias->desc.byteSize];

    F16* ival = (F16*)iptr;
    F16* fval = (F16*)fptr;
    F16* bval = (F16*)bptr;
    for(U32 i = 0; i < input->desc.num; i++){
        ival[i] = (rand() & 1023) / 1024.0 - 0.5;
        U32 s0 = input->desc.stride[0];
        U32 s1 = input->desc.stride[1];
        U32 j  = i % (s0 * s1); 
        if((j % s0) == 0 || (j % s0) == s0 - 1) ival[i] = 0;
        if( j / s0  == 0 ||  j / s0  == s1 - 1) ival[i] = 0;
    }

    for(U32 i = 0; i < flt->desc.num; i++){
        fval[i] = (rand() & 1023) / 1024.0 - 0.5;
    }

    for(U32 i = 0; i < bias->desc.num; i++){
        bval[i] = (rand() & 1023) / 1024.0 - 0.5;
    }

    CHECK_STATUS(gcl_trans_memory(handle, (void*)iptr, (void*)input, &input->desc.byteSize, HOST_TO_DEVICE_BUF, CL_TRUE));
    CHECK_STATUS(gcl_trans_memory(handle, (void*)fptr, (void*)flt,   &flt->desc.byteSize,   HOST_TO_DEVICE_BUF, CL_TRUE));
    CHECK_STATUS(gcl_trans_memory(handle, (void*)bptr, (void*)bias,  &bias->desc.byteSize,  HOST_TO_DEVICE_BUF, CL_TRUE));
#ifdef _DEBUG    
    CHECK_STATUS(gcl_print_memory<F16>(handle, input, "input"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, flt,   "flt"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, bias,  "bias"));
#endif    


    Kernel kernel;
    char kernelname[128];
    for(int i = 0; i < 1; i++){
        sprintf(kernelname, "sample");
        CHECK_STATUS(gcl_create_kernel_binary(handle, kernelname, &kernel));
        U32 iw_str = input->desc.stride[0];
        U32 ih_str = input->desc.stride[1];
        U32 iwh_str = iw_str * ih_str;

        U32 fwh_str = flt->desc.stride[0];
        U32 fc_str  = flt->desc.stride[1]; 
        U32 flt_str = fwh_str * fc_str;

        U32 ow_str = output->desc.stride[0];
        U32 oh_str = output->desc.stride[1];
        U32 oc_str = output->desc.stride[2];
        U32 gs[3];
        gs[0] = ow_str;
        gs[1] = oh_str;
        gs[2] = oc_str;
        U32 dim   = 3;
        U32 ls[3] = {0, 0, 0};
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, iwh_str, fc_str, flt_str, ow_str, oh_str, gs[0], gs[1], input->mem, flt->mem, bias->mem, output->mem));
        CHECK_STATUS(gcl_set_kernelVec(handle, kernel, dim, gs, ls, "sample"));
        CHECK_STATUS(gcl_run_kernelVec(handle));

    }

#ifdef _DEBUG    
    CHECK_STATUS(gcl_print_memory<F16>(handle, output, "output"));
#endif    
    delete[] iptr;
    delete[] fptr;
    delete[] bptr;
    gcl_destroy_gclmem(input);
    gcl_destroy_gclmem(flt);
    gcl_destroy_gclmem(bias);
    gcl_destroy_gclmem(output);
    gcl_destroy_handle(handle);
    }
}



