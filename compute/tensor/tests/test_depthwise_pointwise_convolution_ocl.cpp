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
#include "libkernelsource.h"

#ifdef _USE_FP16
inline GCLMem_t alloc(Tensor tensor)
{
    auto mem = (OclMemory *)tensor.get_memory();
    mem->alloc();
    return (GCLMem_t)mem->get_ptr();
}

inline GCLMem_t alloc_map(Tensor tensor)
{
    auto mem = (OclMemory *)tensor.get_memory();
    mem->mapped_alloc();
    return (GCLMem_t)mem->get_ptr();
}

inline GCLMem_t alloc_bytes(Tensor tensor, U32 size)
{
    auto mem = (OclMemory *)tensor.get_memory();
    GCLMem_t ptr = NULL;
    if (size > 0) {
        mem->resize(tensor1d(DT_U8, size));
        mem->alloc();
        ptr = (GCLMem_t)mem->get_ptr();
    }
    return ptr;
}

inline GCLMem_t alloc_desc(Tensor tensor, GCLMemDesc desc)
{
    auto mem = (OclMemory *)tensor.get_memory();
    mem->padding(desc);
    mem->alloc();
    return (GCLMem_t)mem->get_ptr();
}

int depthwisePointwiseConvolutionTest(
    int argc, char *argv[], DataFormat filterDataFormat, DataType dt)
{
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 group, stride, padding, dilation;
    U32 on, oc, oh, ow;
    U32 biasNum;
    ArchInfo archInfo;
    archInfo.arch = MALI;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;

    in = 1;
    ic = 8;
    ih = 4;
    iw = 4;
    fn = 8;
    fh = 3;
    fw = 3;
    group = 1;
    stride = 1;
    padding = 1;
    dilation = 1;

    if (argc == 9 || argc == 10) {
        ic = atoi(argv[1]);
        ih = atoi(argv[2]);
        iw = atoi(argv[3]);
        fn = atoi(argv[4]);
        fh = atoi(argv[5]);
        fw = atoi(argv[6]);
        stride = atoi(argv[7]);
        padding = atoi(argv[8]);
        if (argc == 10) {
            dilation = atoi(argv[9]);
        }
    }
    fc = ic;
    U32 fhd = (fh - 1) * dilation + 1;
    U32 fwd = (fw - 1) * dilation + 1;
    on = 1;
    oc = fn;
    oh = (ih + padding * 2 - fhd) / stride + 1;
    ow = (iw + padding * 2 - fwd) / stride + 1;
    ActivationParamSpec dwActivationParamSpec;
    ActivationParamSpec pwActivationParamSpec;
    dwActivationParamSpec.mode = ACTIVATION_NULL;
    pwActivationParamSpec.mode = ACTIVATION_NULL;
    ConvolutionParamSpec convParamSpec = createConvolutionParamSpec(group, 1, fh, fw, 1, stride,
        stride, 0, 0, padding, padding, padding, padding, dilation, dilation, dilation, fn,
        Convolution_Depthwise_Pointwise);

    U32 dwFilterLen = 1 * fc * fh * fw;
    U32 pwFilterLen = fn * fc * 1 * 1;
    U32 dwBiasLen = fc;
    U32 pwBiasLen = fn;

    TensorDesc inputDesc = tensor4df(dt, DF_NCHW, in, ic, ih, iw);
    TensorDesc dwFilterDesc = tensor4df(dt, filterDataFormat, 1, fc, fh, fw);
    TensorDesc pwFilterDesc = tensor4df(dt, filterDataFormat, fn, fc, 1, 1);
    TensorDesc dwBiasDesc = tensor1d(dt, dwBiasLen);
    TensorDesc pwBiasDesc = tensor1d(dt, pwBiasLen);

    U8 *input_cpu = ut_input_v(in * ic * ih * iw, dt, UT_INIT_RANDOM);
    U8 *dw_filter_cpu = ut_input_v(dwFilterLen, dt, UT_INIT_RANDOM);
    U8 *pw_filter_cpu = ut_input_v(pwFilterLen, dt, UT_INIT_RANDOM);
    U8 *dw_bias_cpu = ut_input_v(dwBiasLen, dt, UT_INIT_RANDOM);
    U8 *pw_bias_cpu = ut_input_v(pwBiasLen, dt, UT_INIT_RANDOM);

    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    Tensor inputTensor = Tensor(OCLMem);
    Tensor outputTensor = Tensor(OCLMem);
    Tensor dwFilterTensorOrg = Tensor(OCLMem);
    Tensor dwFilterTensor = Tensor(OCLMem);
    Tensor pwFilterTensorOrg = Tensor(OCLMem);
    Tensor pwFilterTensor = Tensor(OCLMem);
    Tensor dwBiasTensor = Tensor(OCLMem);
    Tensor pwBiasTensor = Tensor(OCLMem);
    Tensor pwBiasTensorBuf = Tensor(OCLMem);
    Tensor pwBiasTensorImg = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    inputTensor.resize(inputDesc);
    dwFilterTensor.resize(dwFilterDesc);
    dwFilterTensorOrg.resize(dwFilterDesc);
    pwFilterTensor.resize(pwFilterDesc);
    pwFilterTensorOrg.resize(pwFilterDesc);
    dwBiasTensor.resize(dwBiasDesc);
    pwBiasTensor.resize(pwBiasDesc);
    pwBiasTensorBuf.resize(pwBiasDesc);
    pwBiasTensorImg.resize(pwBiasDesc);

    MaliPara maliPara;
    ForwardRunInfoMali runInfo;
    runInfo.algorithm = (I32)(DEPTHWISE_CONVOLUTION_ALGORITHM_NULL);
    maliPara.handle = handle;
    maliPara.forwardRunInfo = &runInfo;
    archInfo.archPara = &maliPara;

    CHECK_STATUS(depthwise_pointwise_convolution_infer_output_size(
        &inputTensor, dwFilterTensor, pwFilterTensor, convParamSpec, &outputTensor, dt, &archInfo));
    ConvolutionPolicy policy = CONVOLUTION_TUNNING;
    DepthwiseConvolutionForwardAlgorithm alg = DEPTHWISE_CONVOLUTION_ALGORITHM_NULL;
    CHECK_STATUS(depthwise_pointwise_convolution_infer_forward_algorithm(inputTensor,
        dwFilterTensor, pwFilterTensor, outputTensor, convParamSpec, policy, &alg, DT_F16,
        dwActivationParamSpec, pwActivationParamSpec, &archInfo));

    U32 maxBytes = 0;
    U32 tmpBytes;
    CHECK_STATUS(depthwise_pointwise_convolution_infer_forward_tmp_bytes(inputTensor,
        dwFilterTensor, pwFilterTensor, outputTensor, convParamSpec, alg, &tmpBytes, &archInfo));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;

    U32 dwBytes;
    U32 pwBytes;
    U32 str[3] = {0, 0, 0};
    U32 off[3] = {0, 0, 0};
    GCLMemDesc filterMemDesc[2];
    filterMemDesc[0] = gcl_mem_desc(str, off, DT_U8, DF_NCWHC4);
    filterMemDesc[1] = gcl_mem_desc(str, off, DT_U8, DF_NCWHC4);
    maliPara.gclmemFilterDesc = filterMemDesc;
    CHECK_STATUS(depthwise_pointwise_convolution_transform_filter_bytes(
        dwFilterTensor, pwFilterTensor, convParamSpec, alg, &dwBytes, &pwBytes, &archInfo));

    GCLMem_t output = alloc_map(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));

    GCLMemDesc desc = gclmem_build_desc();
    biasNum = (oc + 3) / 4;
    desc.memType = GCL_MEM_IMG_1D;
    desc.byteSize = biasNum * 4 * bytesOf(dt);
    desc.stride[0] = biasNum;
    desc.stride[1] = 1;
    desc.stride[2] = 1;
    desc.offset[0] = 0;
    desc.offset[1] = 0;
    desc.offset[2] = 0;
    desc.num = biasNum;
    desc.memFormat = DF_NHWC;
    desc.flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
    desc.host_ptr = pw_bias_cpu;
    alloc_desc(pwBiasTensorImg, desc);

    biasNum = (oc + 7) / 8 * 8;
    desc.memType = GCL_MEM_BUF;
    desc.byteSize = biasNum * bytesOf(dt);
    desc.stride[0] = biasNum;
    desc.stride[1] = 1;
    desc.stride[2] = 1;
    desc.offset[0] = 0;
    desc.offset[1] = 0;
    desc.offset[2] = 0;
    desc.num = biasNum;
    desc.memFormat = DF_NHWC;
    desc.flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
    desc.host_ptr = pw_bias_cpu;
    alloc_desc(pwBiasTensorBuf, desc);

    biasNum = (ic + 3) / 4;
    desc.memType = GCL_MEM_IMG_1D;
    desc.byteSize = biasNum * 4 * bytesOf(dt);
    desc.stride[0] = biasNum;
    desc.stride[1] = 1;
    desc.stride[2] = 1;
    desc.offset[0] = 0;
    desc.offset[1] = 0;
    desc.offset[2] = 0;
    desc.num = biasNum;
    desc.memFormat = DF_NHWC;
    desc.flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
    desc.host_ptr = dw_bias_cpu;
    alloc_desc(dwBiasTensor, desc);

    desc = filterMemDesc[0];
    alloc_desc(dwFilterTensor, desc);
    desc = filterMemDesc[1];
    alloc_desc(pwFilterTensor, desc);

    desc.stride[0] = fw * fh;
    desc.stride[1] = fc;
    desc.stride[2] = 1;
    desc.offset[0] = 0;
    desc.offset[1] = 0;
    desc.offset[2] = 0;
    desc.byteSize = fw * fh * fc * bytesOf(dt);
    desc.num = fw * fh * fc;
    desc.memType = GCL_MEM_BUF;
    desc.memFormat = DF_NCHW;
    desc.flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
    desc.host_ptr = dw_filter_cpu;
    alloc_desc(dwFilterTensorOrg, desc);

    desc.stride[0] = 1;
    desc.stride[1] = fc;
    desc.stride[2] = fn;
    desc.offset[0] = 0;
    desc.offset[1] = 0;
    desc.offset[2] = 0;
    desc.byteSize = fn * fc * bytesOf(dt);
    desc.num = fn * fc;
    desc.memType = GCL_MEM_BUF;
    desc.memFormat = DF_NCHW;
    desc.flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
    desc.host_ptr = pw_filter_cpu;
    alloc_desc(pwFilterTensorOrg, desc);

    tmpBytes = tensorNumBytes(inputDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(depthwise_pointwise_convolution_transform_filter(dwFilterTensorOrg,
        pwFilterTensorOrg, convParamSpec, alg, &dwFilterTensor, &pwFilterTensor, &archInfo));
    pwBiasTensor = (runInfo.algorithm == (I32)(DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM))
        ? pwBiasTensorBuf
        : pwBiasTensorImg;

    CHECK_STATUS(ocl_set_input(handle, input, inputDesc, input_cpu, tmpbuf, true));

    CHECK_STATUS(depthwise_pointwise_convolution(inputTensor, dwFilterTensor, pwFilterTensor,
        convParamSpec, alg, dwBiasTensor, pwBiasTensor, tmpTensor, outputTensor,
        dwActivationParamSpec, pwActivationParamSpec, &archInfo));
    /*warp up*/
    UNI_INFO_LOG("warm up gpu:\n")
    for (U32 i = 0; i < 2; i++) {
        CHECK_STATUS(gcl_run_kernelVec(handle));
    }

#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernelVec_timing(handle, 0, handle->kernelVec->size()));
    double time = handle->t_execute * 0.001;
#else
    CHECK_STATUS(gcl_run_kernelVec(handle));
#endif
    TensorDesc outputDesc = outputTensor.get_desc();
    CHECK_STATUS(ocl_get_output(handle, output, outputDesc, true));
    void *output_gpu = output->mapPtrArray.back();
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u)+(%u %u %u %u)/(%u %u)=(%u %u %u %u)", in, ic, ih, iw, fn, fc, fh,
        fw, stride, padding, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "DepthwisePointwise", params);
#ifdef _DEBUG
    double ops = 2.0 * in * ic * ih * iw * fh * fw + in * ic * oh * ow +
        2.0 * on * oc * oh * ow * ic + on * oc * oh * ow;
    ut_log(dt, buffer, ops, time);
#endif
    Tensor inputTensorCpu;
    inputTensorCpu.resize(inputDesc);
    inputTensorCpu.alloc();
    memcpy(get_ptr_from_tensor(inputTensorCpu, UT_ARCH), input_cpu, tensorNumBytes(inputDesc));

    Tensor dwFilterTensorCpu;
    dwFilterTensorCpu.resize(dwFilterDesc);
    dwFilterTensorCpu.alloc();
    memcpy(get_ptr_from_tensor(dwFilterTensorCpu, UT_ARCH), dw_filter_cpu,
        tensorNumBytes(dwFilterDesc));

    Tensor pwFilterTensorCpu;
    pwFilterTensorCpu.resize(pwFilterDesc);
    pwFilterTensorCpu.alloc();
    memcpy(get_ptr_from_tensor(pwFilterTensorCpu, UT_ARCH), pw_filter_cpu,
        tensorNumBytes(pwFilterDesc));

    Tensor dwBiasTensorCpu;
    dwBiasTensorCpu.resize(dwBiasDesc);
    dwBiasTensorCpu.alloc();
    memcpy(get_ptr_from_tensor(dwBiasTensorCpu, UT_ARCH), dw_bias_cpu, tensorNumBytes(dwBiasDesc));

    Tensor pwBiasTensorCpu;
    pwBiasTensorCpu.resize(pwBiasDesc);
    pwBiasTensorCpu.alloc();
    memcpy(get_ptr_from_tensor(pwBiasTensorCpu, UT_ARCH), pw_bias_cpu, tensorNumBytes(pwBiasDesc));

    Tensor outputTensorCpu;
    outputTensorCpu.resize(outputDesc);
    outputTensorCpu.alloc();

    Tensor tmpTensorCpu;
    // setup tmp
    CHECK_STATUS(
        depthwise_pointwise_convolution_infer_forward_tmp_bytes(inputTensorCpu, dwFilterTensorCpu,
            pwFilterTensorCpu, outputTensorCpu, convParamSpec, alg, &tmpBytes, &archInfo));
    tmpTensorCpu.resize(tensor1d(DT_F16, tmpBytes / bytesOf(DT_F16)));
    tmpTensorCpu.alloc();

    CHECK_STATUS(depthwise_pointwise_convolution(inputTensorCpu, dwFilterTensorCpu,
        pwFilterTensorCpu, convParamSpec, DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT, dwBiasTensorCpu,
        pwBiasTensorCpu, tmpTensorCpu, outputTensorCpu, dwActivationParamSpec,
        pwActivationParamSpec, &archInfo_org));
    ut_check_a(output_gpu, get_ptr_from_tensor(outputTensorCpu, UT_ARCH), on * oc * ow * oh, dt);

    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(input_cpu);
    free(dw_filter_cpu);
    free(pw_filter_cpu);
    free(dw_bias_cpu);
    free(pw_bias_cpu);
    return 0;
}
#endif

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    depthwisePointwiseConvolutionTest(argc, argv, DF_NCHW, DT_F16);
#endif
    return 0;
}
