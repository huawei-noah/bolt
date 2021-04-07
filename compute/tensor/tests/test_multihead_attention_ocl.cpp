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
int multiheadAttentionTest(int argc, char *argv[], DataType dt)
{
    U32 in, ic, ih, iw;
    U32 fn[4];
    U32 fc[4];
    U32 on, oc, oh, ow;
    U32 firstFCSliceNum[3];
    U32 matmulSliceLen;
    float multiplyAlpha;
    float multiplyBeta;
    std::vector<bool> eltwiseWithLayerNormIn;
    ArchInfo archInfo;
    archInfo.arch = MALI;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;

    in = 1;
    ic = 312;
    ih = 9;
    iw = 1;

    fn[0] = 936;
    fc[0] = 312;

    fn[1] = 312;
    fc[1] = 312;

    fn[2] = 1200;
    fc[2] = 312;

    fn[3] = 312;
    fc[3] = 1200;

    firstFCSliceNum[0] = 312;
    firstFCSliceNum[1] = 312;
    firstFCSliceNum[2] = 312;

    matmulSliceLen = 26;
    multiplyAlpha = 0.196116134524;
    multiplyBeta = 0;
    U32 filterNum = 4;
    U32 lnNum = 2;
    for (U32 i = 0; i < lnNum; ++i) {
        eltwiseWithLayerNormIn.push_back(false);
    }

    if (argc == 20) {
        in = atoi(argv[1]);
        ic = atoi(argv[2]);
        ih = atoi(argv[3]);
        iw = atoi(argv[4]);
        fn[0] = atoi(argv[5]);
        fc[0] = atoi(argv[6]);
        fn[1] = atoi(argv[7]);
        fc[1] = atoi(argv[8]);
        fn[2] = atoi(argv[9]);
        fc[2] = atoi(argv[10]);
        fn[3] = atoi(argv[11]);
        fc[3] = atoi(argv[12]);
        firstFCSliceNum[0] = atoi(argv[13]);
        firstFCSliceNum[1] = atoi(argv[14]);
        firstFCSliceNum[2] = atoi(argv[15]);
        matmulSliceLen = atoi(argv[16]);
        multiplyAlpha = atof(argv[17]);
        multiplyBeta = atof(argv[18]);
        eltwiseWithLayerNormIn[0] = atoi(argv[19]);
        eltwiseWithLayerNormIn[1] = atoi(argv[19]);
    }
    on = 1;
    oc = fn[3];
    oh = ih;
    ow = 1;

    TensorDesc inputDesc, outputDesc;
    std::vector<TensorDesc> filterDesc;
    std::vector<TensorDesc> biasDesc;
    std::vector<TensorDesc> lnAlphaDesc;
    std::vector<TensorDesc> lnBetaDesc;

    inputDesc = tensor3df(dt, DF_MKT, in, ic, ih);
    for (U32 i = 0; i < filterNum; ++i) {
        TensorDesc tmpFilterDesc = tensor4df(dt, DF_NCHW, fn[i], fc[i], 1, 1);
        TensorDesc tmpBiasDesc = tensor1d(dt, fn[i] + 8);
        filterDesc.push_back(tmpFilterDesc);
        biasDesc.push_back(tmpBiasDesc);
    }

    for (U32 i = 0; i < lnNum; ++i) {
        TensorDesc tmpDesc = tensor1d(dt, (ic + 3) / 4 * 4);
        if (i == 1) {
            tmpDesc = tensor1d(dt, (fn[1] + 3) / 4 * 4);
        }
        lnAlphaDesc.push_back(tmpDesc);
        lnBetaDesc.push_back(tmpDesc);
    }

    std::vector<U8 *> filter_cpu;
    std::vector<U8 *> bias_cpu;
    std::vector<U8 *> lnAlpha_cpu;
    std::vector<U8 *> lnBeta_cpu;

    U8 *input_cpu = ut_input_v(in * ic * ih * iw, dt, UT_INIT_RANDOM);

    for (U32 i = 0; i < filterNum; i++) {
        U8 *fltval = ut_input_v(tensorNumElements(filterDesc[i]), dt, UT_INIT_RANDOM);
        U8 *biasval = ut_input_v(tensorNumElements(biasDesc[i]), dt, UT_INIT_RANDOM);
        filter_cpu.push_back(fltval);
        bias_cpu.push_back(biasval);
    }

    for (U32 i = 0; i < lnNum; i++) {
        U8 *alphaVal = ut_input_v(tensorNumElements(lnAlphaDesc[i]), dt, UT_INIT_RANDOM);
        U8 *betaVal = ut_input_v(tensorNumElements(lnBetaDesc[i]), dt, UT_INIT_RANDOM);
        lnAlpha_cpu.push_back(alphaVal);
        lnBeta_cpu.push_back(betaVal);
    }

    U8 *output_gpu = NULL;

    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    Tensor inputTensor = Tensor(OCLMem);
    inputTensor.resize(inputDesc);

    std::vector<Tensor> filterTensorOrg;
    std::vector<Tensor> filterTensor;
    std::vector<Tensor> biasTensor;
    for (U32 i = 0; i < filterNum; i++) {
        Tensor tensor = Tensor(OCLMem);
        tensor.resize(filterDesc[i]);
        filterTensor.push_back(tensor);
        filterTensorOrg.push_back(tensor);
        tensor.resize(biasDesc[i]);
        biasTensor.push_back(tensor);
    }

    std::vector<Tensor> lnAlphaTensor;
    std::vector<Tensor> lnBetaTensor;
    for (U32 i = 0; i < lnNum; i++) {
        Tensor tensor = Tensor(OCLMem);
        tensor.resize(lnAlphaDesc[i]);
        lnAlphaTensor.push_back(tensor);
        tensor.resize(lnBetaDesc[i]);
        lnBetaTensor.push_back(tensor);
    }
    Tensor tmpTensor = Tensor(OCLMem);
    Tensor outputTensor = Tensor(OCLMem);

    MaliPara maliPara;
    ForwardRunInfoMali runInfo;
    runInfo.algorithm = (I32)(CONVOLUTION_ALGORITHM_NULL);
    for (U32 i = 0; i < 6; ++i) {
        runInfo.best_w[i] = 1;
        runInfo.best_c[i] = 1;
        runInfo.best_k[i] = 1;
    }
    maliPara.handle = handle;
    maliPara.forwardRunInfo = &runInfo;
    archInfo.archPara = &maliPara;
    ActivationMode activation = ACTIVATION_GELU;
    CHECK_STATUS(multihead_attention_infer_output_size(
        &inputTensor, filterTensor, &outputTensor, firstFCSliceNum, &archInfo));

    CHECK_STATUS(multihead_attention_infer_forward_algorithm(inputTensor, filterTensor,
        &multiplyAlpha, &multiplyBeta, firstFCSliceNum, matmulSliceLen, eltwiseWithLayerNormIn,
        activation, outputTensor, &archInfo));
    U32 maxBytes = 0;
    U32 tmpBytes;
    CHECK_STATUS(multihead_attention_infer_forward_tmp_bytes(inputTensor, filterTensor,
        eltwiseWithLayerNormIn, firstFCSliceNum, matmulSliceLen, &tmpBytes, &archInfo));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;

    U32 ftmBytes = 0;
    GCLMemDesc filterMemDesc[4];
    U32 stride[3] = {0, 0, 0};
    U32 offset[3] = {0, 0, 0};
    for (U32 i = 0; i < filterNum; i++) {
        filterMemDesc[i] = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
    }
    maliPara.gclmemFilterDesc = filterMemDesc;
    CHECK_STATUS(multihead_attention_transform_filter_bytes(filterTensor, &ftmBytes, &archInfo));

    GCLMem_t output = alloc_map(outputTensor);

    for (U32 i = 0; i < 2; ++i) {
        U32 biasNum = fn[i] + 8;
        GCLMemDesc tmpDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        tmpDesc.stride[0] = biasNum;
        tmpDesc.stride[1] = 1;
        tmpDesc.stride[2] = 1;
        tmpDesc.offset[0] = 0;
        tmpDesc.offset[1] = 0;
        tmpDesc.offset[2] = 0;
        tmpDesc.num = biasNum;
        tmpDesc.byteSize = biasNum * bytesOf(dt);
        tmpDesc.memFormat = DF_NHWC;
        tmpDesc.memType = GCL_MEM_BUF;
        tmpDesc.flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
        tmpDesc.host_ptr = bias_cpu[i];
        alloc_desc(biasTensor[i], tmpDesc);
    }
    for (U32 i = 2; i < filterNum; ++i) {
        U32 biasNum = (fn[i] + 3) / 4;
        GCLMemDesc tmpDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        tmpDesc.stride[0] = biasNum;
        tmpDesc.stride[1] = 1;
        tmpDesc.stride[2] = 1;
        tmpDesc.offset[0] = 0;
        tmpDesc.offset[1] = 0;
        tmpDesc.offset[2] = 0;
        tmpDesc.num = biasNum;
        tmpDesc.byteSize = biasNum * bytesOf(dt) * 4;
        tmpDesc.memFormat = DF_NHWC;
        tmpDesc.memType = GCL_MEM_IMG_1D;
        tmpDesc.flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
        tmpDesc.host_ptr = bias_cpu[i];
        alloc_desc(biasTensor[i], tmpDesc);
    }

    for (U32 i = 0; i < lnNum; ++i) {
        U32 layerNormNum = (ic + 3) / 4 * 4;
        if (i == 1) {
            layerNormNum = (fn[1] + 3) / 4 * 4;
        }
        GCLMemDesc tmpDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        tmpDesc.stride[0] = layerNormNum;
        tmpDesc.stride[1] = 1;
        tmpDesc.stride[2] = 1;
        tmpDesc.offset[0] = 0;
        tmpDesc.offset[1] = 0;
        tmpDesc.offset[2] = 0;
        tmpDesc.num = layerNormNum;
        tmpDesc.byteSize = layerNormNum * bytesOf(dt);
        tmpDesc.memFormat = DF_NHWC;
        tmpDesc.memType = GCL_MEM_BUF;
        tmpDesc.flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
        tmpDesc.host_ptr = lnAlpha_cpu[i];
        alloc_desc(lnAlphaTensor[i], tmpDesc);

        tmpDesc.stride[0] = layerNormNum;
        tmpDesc.stride[1] = 1;
        tmpDesc.stride[2] = 1;
        tmpDesc.offset[0] = 0;
        tmpDesc.offset[1] = 0;
        tmpDesc.offset[2] = 0;
        tmpDesc.num = layerNormNum;
        tmpDesc.byteSize = layerNormNum * bytesOf(dt);
        tmpDesc.memFormat = DF_NHWC;
        tmpDesc.memType = GCL_MEM_BUF;
        tmpDesc.flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
        tmpDesc.host_ptr = lnBeta_cpu[i];
        alloc_desc(lnBetaTensor[i], tmpDesc);
    }
    for (U32 i = 0; i < filterNum; ++i) {
        GCLMemDesc desc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        desc.stride[0] = fc[i];
        desc.stride[1] = fn[i];
        desc.stride[2] = 1;
        desc.offset[0] = 0;
        desc.offset[1] = 0;
        desc.offset[2] = 0;
        desc.byteSize = fc[i] * fn[i] * bytesOf(dt);
        desc.num = fc[i] * fn[i];
        desc.memType = GCL_MEM_BUF;
        desc.memFormat = DF_NCHW;
        desc.flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
        desc.host_ptr = filter_cpu[i];
        alloc_desc(filterTensorOrg[i], desc);
    }

    for (U32 i = 0; i < filterNum; ++i) {
        GCLMemDesc desc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        desc = filterMemDesc[i];
        alloc_desc(filterTensor[i], desc);
    }

    auto inputMem = (OclMemory *)inputTensor.get_memory();
    GCLMemDesc inputGclDesc = inputMem->get_desc();
    inputGclDesc.flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
    inputGclDesc.host_ptr = input_cpu;
    alloc_desc(inputTensor, inputGclDesc);

    tmpBytes = tensorNumBytes(inputDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    alloc_bytes(tmpTensor, maxBytes);

    std::vector<Tensor *> filterTensorPtr;
    for (U32 i = 0; i < filterNum; i++) {
        filterTensorPtr.push_back(&filterTensor[i]);
    }
    CHECK_STATUS(multihead_attention_transform_filter(filterTensorOrg, filterTensorPtr, &archInfo));
    CHECK_STATUS(multihead_attention(inputTensor, filterTensor, biasTensor, lnAlphaTensor,
        lnBetaTensor, &multiplyAlpha, &multiplyBeta, firstFCSliceNum, matmulSliceLen,
        eltwiseWithLayerNormIn, activation, tmpTensor, outputTensor, &archInfo));
    /*warp up*/
    UNI_INFO_LOG("warm up gpu:\n")
    for (U32 i = 0; i < 2; i++) {
        CHECK_STATUS(gcl_run_kernelVec(handle));
    }
    UNI_INFO_LOG("Run:\n")
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernelVec_timing(handle, 0, handle->kernelVec->size()));
//    double time = handle->t_execute * 0.001;
#else
    CHECK_STATUS(gcl_run_kernelVec(handle));
#endif
    outputDesc = outputTensor.get_desc();
    CHECK_STATUS(ocl_get_output(handle, output, outputDesc, true));
    output_gpu = output->mapPtrArray.back();

    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(input_cpu);
    for (auto p : filter_cpu) {
        free((U8 *)p);
    }
    for (auto p : bias_cpu) {
        free((U8 *)p);
    }
    for (auto p : lnAlpha_cpu) {
        free((U8 *)p);
    }
    for (auto p : lnBeta_cpu) {
        free((U8 *)p);
    }
    return 0;
}
#endif

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    multiheadAttentionTest(argc, argv, DT_F16);
#endif
    return 0;
}
