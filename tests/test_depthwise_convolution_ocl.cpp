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

int depthwiseConvolutionTest(int argc, char* argv[], DataFormat filterDataFormat, DataType dt)
{
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 stride, padding;
    U32 on, oc, oh, ow;
    U32 biasNum;
    Arch arch = MALI;
   
    in = 1;
    ic = 8;
    ih = 4;
    iw = 4;
    fn = 8;
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

    if(filterDataFormat == DF_NCHW) {
        if(fn != ic) {
            std::cout << "ignored depthwise convolution for para fn != ic" << std::endl; 
            return 0;
        }
        fc = 1;
    } else {
        fc = ic;
    }


    on = 1;
    oc = fn;
    oh = (ih + padding * 2 - fh) / stride + 1;
    ow = (iw + padding * 2 - fw) / stride + 1;
    ActivationDesc dwActivationDesc;
    ActivationDesc pwActivationDesc;
    dwActivationDesc.mode = ACTIVATION_NULL;
    pwActivationDesc.mode = ACTIVATION_NULL;

    TensorDesc inputDesc, filterDesc, outputDesc, biasDesc;
    ConvolutionDesc convDesc;
    convDesc.stride_h = stride;
    convDesc.stride_w = stride;
    convDesc.padding_top = padding;
    convDesc.padding_bottom = padding;
    convDesc.padding_left = padding;
    convDesc.padding_right = padding;
    convDesc.dilatedRate_h = 1;
    convDesc.dilatedRate_w = 1;

    U32 filterLen = fn * fc * fh * fw;
    U32 biasLen = oc;
    if(filterDataFormat == DF_CHW_NC) {
       filterLen = fc * fh * fw + fn * fc;
       biasLen   = ic + oc;
    }
    
    inputDesc  = tensor4df(dt, DF_NCHW, in, ic, ih, iw);
    filterDesc = tensor4df(dt, filterDataFormat, fn, fc, fh, fw);
    biasDesc   = tensor1d(dt, biasLen);
    U8 *input_cpu  = ut_input_v(in*ic*ih*iw, dt, UT_INIT_RANDOM);
    U8 *filter_cpu = ut_input_v(filterLen, dt, UT_INIT_RANDOM);
    U8 *bias_cpu   = ut_input_v(biasLen, dt, UT_INIT_RANDOM);
    U8 *output_cpu = ut_input_v(on*oc*oh*ow, dt, UT_INIT_ZERO);
    //U8 *output_gpu = ut_input_v(on*oc*oh*ow, dt, UT_INIT_ZERO);
    U8 *output_gpu = NULL;
    

    GCLHandle_t handle;
    CHECK_STATUS(gcl_create_handle(&handle));
    CHECK_STATUS(gcl_regist_binMap(handle));

    ExtInfo extInfo;
    U32 str[3] = {0, 0, 0};
    U32 off[3] = {0, 0, 0};
    GCLMemDesc inputMemDesc  = gcl_mem_desc(str, off, DT_U8, DF_NCWHC4);
    GCLMemDesc outputMemDesc = gcl_mem_desc(str, off, DT_U8, DF_NCWHC4);
    GCLMemDesc filterMemDesc[2];
    filterMemDesc[0] = gcl_mem_desc(str, off, DT_U8, DF_NCWHC4);
    filterMemDesc[1] = gcl_mem_desc(str, off, DT_U8, DF_NCWHC4);
    ForwardRunInfoMali runInfo;
    runInfo.algorithm = (I32)(DEPTHWISE_CONVOLUTION_ALGORITHM_NULL);
    extInfo.maliInfo.handle = handle;
    extInfo.maliInfo.gclmemInputDesc  = NULL;
    extInfo.maliInfo.gclmemOutputDesc = NULL;
    extInfo.maliInfo.gclmemFilterDesc = NULL;
    extInfo.maliInfo.forwardRunInfo   = &runInfo;
    
    U32 outputBytes;
    CHECK_STATUS(depthwise_convolution_infer_output_size(inputDesc, filterDesc, convDesc, &outputDesc, dt, &outputBytes, arch, &extInfo));
    ConvolutionPolicy policy = CONVOLUTION_TUNNING;
    DepthwiseConvolutionForwardAlgorithm alg = DEPTHWISE_CONVOLUTION_ALGORITHM_NULL;    
    CHECK_STATUS(depthwise_convolution_infer_forward_algorithm(inputDesc, filterDesc, outputDesc, convDesc, policy, &alg, dt, dwActivationDesc, pwActivationDesc, arch, &extInfo));

    extInfo.maliInfo.gclmemInputDesc  = &inputMemDesc;
    extInfo.maliInfo.gclmemOutputDesc = &outputMemDesc;
    extInfo.maliInfo.gclmemFilterDesc = filterMemDesc;
    CHECK_STATUS(depthwise_convolution_infer_output_size(inputDesc, filterDesc, convDesc, NULL, dt, NULL, arch, &extInfo));

    U32 maxBytes = 0;
    U32 tmpBytes;
    CHECK_STATUS(depthwise_convolution_infer_forward_tmp_bytes(inputDesc, filterDesc, outputDesc, convDesc, alg, &tmpBytes, arch, &extInfo));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;

    U32 ftmBytes;
    CHECK_STATUS(depthwise_convolution_transform_filter_bytes(filterDesc, alg, &ftmBytes, arch, &extInfo));

    GCLMem_t input      = gcl_create_gclmem();
    GCLMem_t filter     = gcl_create_gclmem();
    GCLMem_t filter_dp  = gcl_create_gclmem();
    GCLMem_t output     = gcl_create_gclmem();
    GCLMem_t bias       = gcl_create_gclmem();
    GCLMem_t bias_dp    = gcl_create_gclmem();
    GCLMem_t bias_buf   = gcl_create_gclmem();
    GCLMem_t tmpbuf     = gcl_create_gclmem();
    GCLMem_t filter_org    = gcl_create_gclmem();
    GCLMem_t filter_org_dp = gcl_create_gclmem();

    outputMemDesc.use_map = true;
    outputMemDesc.flags   =  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
    outputMemDesc.byteSize = 2 * outputMemDesc.byteSize;
    input->desc  = inputMemDesc;
    filter->desc = filterMemDesc[0];
    filter_dp->desc = filterMemDesc[1];
    output->desc = outputMemDesc;

    biasNum = (oc + 3) / 4 ;
    if(filterDataFormat == DF_CHW_NC) {
        bias_dp->desc.memType    = GCL_MEM_IMG_1D;
        bias_dp->desc.byteSize   = biasNum * 4 * bytesOf(dt);
        bias_dp->desc.stride[0]  = biasNum;
        bias_dp->desc.stride[1]  = 1;
        bias_dp->desc.stride[2]  = 1;
        bias_dp->desc.offset[0]  = 0;
        bias_dp->desc.offset[1]  = 0;
        bias_dp->desc.offset[2]  = 0;
        bias_dp->desc.num        = biasNum;
        bias_dp->desc.memFormat  = DF_NHWC;
        bias_dp->desc.flags      = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
        bias_dp->desc.host_ptr   = bias_cpu + ic * bytesOf(dt);
        gcl_create_memory(handle, bias_dp);
    }

    biasNum = (oc + 7) / 8 * 8;
    if(filterDataFormat == DF_CHW_NC) {
        bias_buf->desc.memType    = GCL_MEM_BUF;
        bias_buf->desc.byteSize   = biasNum * bytesOf(dt);
        bias_buf->desc.stride[0]  = biasNum;
        bias_buf->desc.stride[1]  = 1;
        bias_buf->desc.stride[2]  = 1;
        bias_buf->desc.offset[0]  = 0;
        bias_buf->desc.offset[1]  = 0;
        bias_buf->desc.offset[2]  = 0;
        bias_buf->desc.num        = biasNum;
        bias_buf->desc.memFormat  = DF_NHWC;
        bias_buf->desc.flags      = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
        bias_buf->desc.host_ptr   = bias_cpu + ic * bytesOf(dt);
        biasNum = (ic + 3) / 4;
        gcl_create_memory(handle, bias_buf);
    }

    bias->desc.memType    = GCL_MEM_IMG_1D;
    bias->desc.byteSize   = biasNum * 4 * bytesOf(dt);
    bias->desc.stride[0]  = biasNum;
    bias->desc.stride[1]  = 1;
    bias->desc.stride[2]  = 1;
    bias->desc.offset[0]  = 0;
    bias->desc.offset[1]  = 0;
    bias->desc.offset[2]  = 0;
    bias->desc.num        = biasNum;
    bias->desc.memFormat  = DF_NHWC;
    bias->desc.flags      = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
    bias->desc.host_ptr   = bias_cpu;

    
    if(filterDataFormat == DF_CHW_NC) {
        filter_org->desc.stride[0]  = fw * fh;
        filter_org->desc.stride[1]  = 1;
        filter_org->desc.stride[2]  = fc;
        filter_org->desc.offset[0]  = 0;
        filter_org->desc.offset[1]  = 0;
        filter_org->desc.offset[2]  = 0;
        filter_org->desc.byteSize   = fw * fh * fc * bytesOf(dt);
        filter_org->desc.num        = fw * fh * fc;
        filter_org->desc.memType    = GCL_MEM_BUF;
        filter_org->desc.memFormat  = DF_NCHW;
        filter_org->desc.flags      = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
        filter_org->desc.host_ptr   = filter_cpu;

        filter_org_dp->desc.stride[0]  = 1;
        filter_org_dp->desc.stride[1]  = fc;
        filter_org_dp->desc.stride[2]  = fn;
        filter_org_dp->desc.offset[0]  = 0;
        filter_org_dp->desc.offset[1]  = 0;
        filter_org_dp->desc.offset[2]  = 0;
        filter_org_dp->desc.byteSize   = fn * fc * bytesOf(dt);
        filter_org_dp->desc.num        = fn * fc;
        filter_org_dp->desc.memType    = GCL_MEM_BUF;
        filter_org_dp->desc.memFormat  = DF_NCHW;
        filter_org_dp->desc.flags      = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
        filter_org_dp->desc.host_ptr   = filter_cpu + fw * fh * fc * bytesOf(dt);
        gcl_create_memory(handle, filter_org_dp);
        gcl_create_memory(handle, filter_dp);
    } else {
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
    }

    gcl_create_memory(handle, input);
    gcl_create_memory(handle, output);
    gcl_create_memory(handle, filter);
    gcl_create_memory(handle, bias);
    gcl_create_memory(handle, filter_org);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));

    CHECK_STATUS(tensor_computing_set_input_infer_tmpBuf_size(input, inputDesc, &tmpBytes, arch));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    CHECK_STATUS(tensor_computing_get_output_infer_tmpBuf_size(output, outputDesc, &tmpBytes, arch));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpbuf->desc.byteSize = maxBytes;
    if(maxBytes) gcl_create_memory(handle, tmpbuf);

    TensorDesc filterDescTran;
    GCLMem filter_org_array[2];
    filter_org_array[0] = *filter_org;
    filter_org_array[1] = *filter_org_dp;
    GCLMem filter_array[2];
    filter_array[0] = *filter;
    filter_array[1] = *filter_dp;
    CHECK_STATUS(depthwise_convolution_transform_filter(filterDesc, filter_org_array, alg, &filterDescTran, filter_array, arch, &extInfo));
    GCLMem bias_array[2];
    bias_array[0] = *bias;
    bias_array[1] = *bias_dp;
    if(runInfo.algorithm == (I32)(DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM)) bias_array[1] = *bias_buf;


    CHECK_STATUS(tensor_computing_set_input(input, inputDesc, input_cpu, tmpbuf, true, arch, &extInfo));
    CHECK_STATUS(depthwise_convolution(inputDesc, input, filterDesc, filter_array, convDesc, alg,
                 biasDesc, bias_array, tmpBytes, tmpbuf, outputDesc, output, dwActivationDesc, pwActivationDesc, arch, &extInfo));
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
    double time = handle->t_total * 0.001;
    double ops;
    if (filterDataFormat == DF_CHW_NC) {
        sprintf(buffer, "%20s, %80s", "DepthwisePointwise", params);
        ops = 2.0 * in * ic * ih * iw * fh * fw + in * ic * oh * ow +
                 2.0 * on * oc * oh * ow * ic + on * oc * oh * ow;
    }
    if (filterDataFormat == DF_NCHW) {
        sprintf(buffer, "%20s, %80s", "DepthwiseConvolution", params);
        ops = 2.0 * in * ic * ih * iw * fh * fw + in * ic * oh * ow;
    }
    ut_log(dt, buffer, ops, time);
#endif
    CHECK_STATUS(depthwise_convolution(inputDesc, input_cpu, filterDesc, filter_cpu, convDesc, DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT,
                                 biasDesc, bias_cpu, tmpBytes, NULL, outputDesc, output_cpu, dwActivationDesc, pwActivationDesc, CPU_GENERAL));
    ut_check_a(output_gpu, output_cpu, on * oc * ow * oh, dt);

    CHECK_STATUS(gcl_finish(handle));
    free(input_cpu ); 
    free(filter_cpu); 
    free(bias_cpu  ); 
    free(output_cpu); 
//    free(output_gpu);
    CHECK_STATUS(gcl_unmap_memory(handle, output));
    gcl_destroy_gclmem(input);
    gcl_destroy_gclmem(filter);
    gcl_destroy_gclmem(output);
    gcl_destroy_gclmem(bias);
    gcl_destroy_gclmem(tmpbuf);
    gcl_destroy_gclmem(filter_org);
    if(filterDataFormat == DF_CHW_NC) {
        gcl_destroy_gclmem(filter_org_dp);
        gcl_destroy_gclmem(filter_dp);
        gcl_destroy_gclmem(bias_dp);
        gcl_destroy_gclmem(bias_buf);
    }
    gcl_destroy_handle(handle);
    return 0;
}


int main(int argc, char** argv) {
#ifdef _USE_FP16
    depthwiseConvolutionTest(argc, argv, DF_CHW_NC, DT_F16);
    depthwiseConvolutionTest(argc, argv, DF_NCHW,   DT_F16);
#endif
    return 0;
}
