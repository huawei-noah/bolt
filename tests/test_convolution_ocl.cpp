// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <string.h>
#include "tensor_computing.h"
#include "ut_util.h"
#include "gcl.h"
#include "libkernelbin.h"

int convolutionTest(int argc, char* argv[], DataType dt)
{
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 stride, padding;
    U32 on, oc, oh, ow;
    U32 biasNum;
    Arch arch = MALI;
    
    in = 1;
    ic = 4;
    ih = 4;
    iw = 4;
    fn = 4;
    fh = 3;
    fw = 3;
    stride = 1;
    padding = 1;

    if(argc == 9) {
        ic = atoi(argv[1]);
        ih = atoi(argv[2]);
        iw = atoi(argv[3]);
        fn = atoi(argv[4]);
        fh = atoi(argv[5]);
        fw = atoi(argv[6]);
        stride  = atoi(argv[7]);
        padding = atoi(argv[8]);
    }

    fc = ic;

    on = 1;
    oc = fn;
    oh = (ih + padding * 2 - fh) / stride + 1;
    ow = (iw + padding * 2 - fw) / stride + 1;
    ActivationDesc activationDesc;
    activationDesc.mode = ACTIVATION_NULL;

    TensorDesc inputDesc, inputDesc_gpu, filterDesc, outputDesc, biasDesc;
    ConvolutionDesc convDesc;
    convDesc.stride_h = stride;
    convDesc.stride_w = stride;
    convDesc.padding_top = padding;
    convDesc.padding_bottom = padding;
    convDesc.padding_left = padding;
    convDesc.padding_right = padding;
    convDesc.dilatedRate_h = 1;
    convDesc.dilatedRate_w = 1;

    inputDesc  = tensor4df(dt, DF_NCHW, in, ic, ih, iw);
    filterDesc = tensor4df(dt, DF_NCHW, fn, fc, fh, fw);
    biasDesc   = tensor1d(dt, oc);
    U8 *input_cpu  = ut_input_v(in*ic*ih*iw, dt, UT_INIT_RANDOM);
    U8 *filter_cpu = ut_input_v(fn*fc*fh*fw, dt, UT_INIT_RANDOM);
    U8 *bias_cpu   = ut_input_v(oc, dt, UT_INIT_RANDOM);
    U8 *output_cpu = ut_input_v(on*oc*oh*ow, dt, UT_INIT_ZERO);
//    U8 *output_gpu = ut_input_v(on*oc*oh*ow, dt, UT_INIT_ZERO);
    U8 *output_gpu = NULL;
//    inputDesc_gpu  = tensor4df(dt, DF_NCHW_ORG_MALI, in, ic, ih, iw);//first layer test
    inputDesc_gpu  = tensor4df(dt, DF_NCHW, in, ic, ih, iw);

    GCLHandle_t handle;
    CHECK_STATUS(gcl_create_handle(&handle));
    CHECK_STATUS(gcl_regist_binMap(handle));

    ExtInfo extInfo;
    U32 str[3] = {0, 0, 0};
    U32 off[3] = {0, 0, 0};
    GCLMemDesc inputMemDesc  = gcl_mem_desc(str, off, DT_U8, DF_NCWHC4);
    GCLMemDesc outputMemDesc = gcl_mem_desc(str, off, DT_U8, DF_NCWHC4);
    GCLMemDesc filterMemDesc = gcl_mem_desc(str, off, DT_U8, DF_NCWHC4);
    ForwardRunInfoMali runInfo;
    runInfo.algorithm = (I32)(CONVOLUTION_ALGORITHM_NULL);
    extInfo.maliInfo.handle = handle;
    extInfo.maliInfo.gclmemInputDesc  = NULL;
    extInfo.maliInfo.gclmemOutputDesc = NULL;
    extInfo.maliInfo.gclmemFilterDesc = NULL;
    extInfo.maliInfo.forwardRunInfo   = &runInfo;
    
    U32 outputBytes;
    CHECK_STATUS(convolution_infer_output_size(inputDesc_gpu, filterDesc, convDesc, &outputDesc, dt, &outputBytes, arch, &extInfo));

    ConvolutionPolicy policy = CONVOLUTION_TUNNING;
    ConvolutionForwardAlgorithm alg = CONVOLUTION_ALGORITHM_NULL;
    CHECK_STATUS(convolution_infer_forward_algorithm(inputDesc_gpu, filterDesc, outputDesc, convDesc, policy, &alg, dt, activationDesc, arch, &extInfo));

    extInfo.maliInfo.gclmemInputDesc  = &inputMemDesc;
    extInfo.maliInfo.gclmemOutputDesc = &outputMemDesc;
    extInfo.maliInfo.gclmemFilterDesc = &filterMemDesc;
    CHECK_STATUS(convolution_infer_output_size(inputDesc_gpu, filterDesc, convDesc, NULL, dt, NULL, arch, &extInfo));

    U32 maxBytes = 0;
    U32 tmpBytes;
    CHECK_STATUS(convolution_infer_forward_tmp_bytes(inputDesc_gpu, filterDesc, outputDesc, convDesc, alg, &tmpBytes, arch, &extInfo));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;

    U32 ftmBytes;
    CHECK_STATUS(convolution_transform_filter_bytes(filterDesc, alg, &ftmBytes, arch, &extInfo));

    GCLMem_t input      = gcl_create_gclmem();
    GCLMem_t filter     = gcl_create_gclmem();
    GCLMem_t output     = gcl_create_gclmem();
    GCLMem_t bias       = gcl_create_gclmem();
    GCLMem_t tmpbuf     = gcl_create_gclmem();
    GCLMem_t filter_org = gcl_create_gclmem();
    outputMemDesc.use_map = true;
    outputMemDesc.flags   =  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
    outputMemDesc.byteSize = 2 * outputMemDesc.byteSize;
    input->desc  = inputMemDesc;
    filter->desc = filterMemDesc;
    output->desc = outputMemDesc;
    if(fh == 1 && fw == 1 && ih == 1 && iw == 1) {
        biasNum = oc;
        bias->desc.memType  = GCL_MEM_BUF;
        bias->desc.byteSize = biasNum * bytesOf(dt);
    } else {
        biasNum = (oc + 3) / 4;
        bias->desc.memType  = GCL_MEM_IMG_1D;
        bias->desc.byteSize = biasNum * 4 * bytesOf(dt);
    }
    bias->desc.stride[0]  = biasNum;
    bias->desc.stride[1]  = 1;
    bias->desc.stride[2]  = 1;
    bias->desc.offset[0]  = 0;
    bias->desc.offset[1]  = 0;
    bias->desc.offset[2]  = 0;
    bias->desc.num        = biasNum;
    bias->desc.memFormat  = DF_NHWC;
    bias->desc.flags      = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
    U8* bias_cpu_align = NULL;
    if((oc & 3) != 0) {
        U8 *bias_cpu_align = ut_input_v((oc + 3) / 4 * 4, dt, UT_INIT_ZERO);
        memcpy(bias_cpu_align, bias_cpu, (oc + 3) / 4 * 4 * bytesOf(dt));
        bias->desc.host_ptr = bias_cpu_align;
    } else {
        bias->desc.host_ptr   = bias_cpu;
    }


    filter_org->desc.stride[0]  = fw * fh;
    filter_org->desc.stride[1]  = fc;
    filter_org->desc.stride[2]  = fn;
    filter_org->desc.offset[0]  = 0;
    filter_org->desc.offset[1]  = 0;
    filter_org->desc.offset[2]  = 0;
    filter_org->desc.byteSize   = fw * fh * fc * fn * bytesOf(dt);
    filter_org->desc.num        = fw * fh * fc * fn;
    filter_org->desc.memType    = GCL_MEM_BUF;
    filter_org->desc.memFormat  = DF_NCHW;
    filter_org->desc.flags      = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
    filter_org->desc.host_ptr   = filter_cpu;

    gcl_create_memory(handle, input);
    gcl_create_memory(handle, output);
    gcl_create_memory(handle, filter);
    gcl_create_memory(handle, bias);
    gcl_create_memory(handle, filter_org);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));

    CHECK_STATUS(tensor_computing_set_input_infer_tmpBuf_size(input, inputDesc_gpu, &tmpBytes, arch));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    CHECK_STATUS(tensor_computing_get_output_infer_tmpBuf_size(output, outputDesc, &tmpBytes, arch));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpbuf->desc.byteSize = maxBytes;
    if(maxBytes) gcl_create_memory(handle, tmpbuf);

    TensorDesc filterDescTran;
    CHECK_STATUS(convolution_transform_filter(filterDesc, filter_org, alg, &filterDescTran, filter, tmpbuf, arch, &extInfo));

    CHECK_STATUS(tensor_computing_set_input(input, inputDesc_gpu, input_cpu, tmpbuf, true, arch, &extInfo));
    CHECK_STATUS(convolution(inputDesc_gpu, input, filterDesc, filter, convDesc, alg, biasDesc, nullptr,
                 biasDesc, bias, tmpBytes, tmpbuf, outputDesc, output, activationDesc, arch, &extInfo));
#ifndef _DEBUG
    CHECK_STATUS(gcl_run_kernelVec(handle));
#endif
    CHECK_STATUS(tensor_computing_get_output(output, outputDesc, NULL, NULL, true, arch, &extInfo));
    output_gpu = output->desc.map_ptr;
#ifdef _DEBUG    
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u)+(%u %u %u %u)/(%u %u)=(%u %u %u %u)",
                    in, ic, ih, iw,
                    fn, fc, fh, fw,
                    stride, padding,
                    on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "Convolution", params);
    double time = handle->t_total * 0.001;
    double ops = (1.0 * on * oc * oh * ow) * (2.0 * ic * fh * fw + 1);
    ut_log(dt, buffer, ops, time);
#endif
    CHECK_STATUS(convolution(inputDesc, input_cpu, filterDesc, filter_cpu, convDesc, CONVOLUTION_ALGORITHM_GEMM,
                                 biasDesc, nullptr, biasDesc, bias_cpu, tmpBytes, NULL, outputDesc, output_cpu, activationDesc, CPU_GENERAL));
    ut_check_a(output_gpu, output_cpu, on * oc * ow * oh, dt);

    CHECK_STATUS(gcl_finish(handle));
    free(input_cpu); 
    free(filter_cpu); 
    free(bias_cpu); 
    free(output_cpu);
    if(bias_cpu_align) free(bias_cpu_align);
//    free(output_gpu); 
    CHECK_STATUS(gcl_unmap_memory(handle, output));
    gcl_destroy_gclmem(input);
    gcl_destroy_gclmem(filter);
    gcl_destroy_gclmem(output);
    gcl_destroy_gclmem(bias);
    gcl_destroy_gclmem(tmpbuf);
    gcl_destroy_gclmem(filter_org);
    gcl_destroy_handle(handle);
    return 0;
}


int main(int argc, char** argv) {
#ifdef _USE_FP16
    convolutionTest(argc, argv, DT_F16);
#endif
    return 0;
}
