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
int fullyConnectedTest(int argc, char *argv[], DataType dt)
{
    U32 in, ic, ih, iw;
    U32 fn, fc;
    U32 on, oc, oh, ow;
    U32 biasNum;
    ArchInfo archInfo;
    archInfo.arch = MALI;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;

    in = 1;
    ic = 4;
    ih = 4;
    iw = 4;
    fc = 64;
    fn = 4;

    if (argc == 6) {
        ic = atoi(argv[1]);
        ih = atoi(argv[2]);
        iw = atoi(argv[3]);
        fc = atoi(argv[4]);
        fn = atoi(argv[5]);
    }

    if (iw * ih * ic % fc != 0) {
        CHECK_STATUS(NOT_MATCH);
    }
    U32 row = iw * ih * ic / fc;
    on = 1;
    oc = 1;
    oh = row;
    ow = fn;

    TensorDesc inputDesc, filterDesc, outputDesc, biasDesc;
    TensorDesc filterDesc_cpu, outputDesc_cpu;

    inputDesc = tensor2df(dt, DF_NORMAL, row, fc);
    filterDesc = tensor2df(dt, DF_NORMAL, fn, fc);
    outputDesc_cpu = tensor2df(dt, DF_NORMAL, oh, ow);
    biasDesc = tensor1d(dt, fn);

    U8 *input_cpu = ut_input_v(in * ic * ih * iw, dt, UT_INIT_RANDOM);
    U8 *filter_cpu = ut_input_v(fn * fc, dt, UT_INIT_RANDOM);
    U8 *bias_cpu = ut_input_v(fn, dt, UT_INIT_RANDOM);
    U8 *output_gpu = NULL;

    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    Tensor inputTensor = Tensor(OCLMem);
    Tensor outputTensor = Tensor(OCLMem);
    Tensor filterTensorOrg = Tensor(OCLMem);
    Tensor filterTensor = Tensor(OCLMem);
    Tensor biasTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    inputTensor.resize(inputDesc);
    filterTensor.resize(filterDesc);
    filterTensorOrg.resize(filterDesc);
    biasTensor.resize(biasDesc);

    MaliPara maliPara;
    ForwardRunInfoMali runInfo;
    runInfo.algorithm = (I32)(CONVOLUTION_ALGORITHM_NULL);
    runInfo.best_w[0] = 1;
    runInfo.best_c[0] = 1;
    runInfo.best_k[0] = 1;
    maliPara.handle = handle;
    maliPara.forwardRunInfo = &runInfo;
    archInfo.archPara = &maliPara;

    CHECK_STATUS(
        fully_connected_infer_output_size(&inputTensor, filterTensor, &outputTensor, &archInfo));
    CHECK_STATUS(
        fully_connected_infer_forward_algorithm(inputTensor, filterTensor, outputTensor, &archInfo));

    U32 maxBytes = 0;
    U32 tmpBytes;
    CHECK_STATUS(
        fully_connected_infer_forward_tmp_bytes(inputTensor, filterTensor, &tmpBytes, &archInfo));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;

    U32 ftmBytes;
    U32 str[3] = {0, 0, 0};
    U32 off[3] = {0, 0, 0};
    GCLMemDesc filterMemDesc = gcl_mem_desc(str, off, DT_U8, DF_NCHW);
    maliPara.gclmemFilterDesc = &filterMemDesc;
    CHECK_STATUS(fully_connected_transform_filter_bytes(filterTensor, &ftmBytes, &archInfo));
    GCLMem_t output = alloc_map(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));

    U32 item_m = runInfo.best_w[0];
    GCLMemDesc desc = gclmem_build_desc();
    biasNum = (fn + item_m - 1) / item_m * item_m;
    U8 *bias_val = bias_cpu;
    if (biasNum > fn) {
        bias_val = ut_input_v(biasNum, dt, UT_INIT_ZERO);
        memcpy(bias_val, bias_cpu, fn * bytesOf(dt));
    }
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
    desc.host_ptr = bias_val;
    GCLMem_t bias = alloc_desc(biasTensor, desc);
    if (biasNum > fn) {
        free(bias_val);
    }

    desc = filterMemDesc;
    GCLMem_t filter = alloc_desc(filterTensor, desc);

    desc.stride[0] = fc;
    desc.stride[1] = fn;
    desc.stride[2] = 1;
    desc.offset[0] = 0;
    desc.offset[1] = 0;
    desc.offset[2] = 0;
    desc.byteSize = fc * fn * bytesOf(dt);
    desc.num = fc * fn;
    desc.memType = GCL_MEM_BUF;
    desc.memFormat = DF_NCHW;
    desc.flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
    desc.host_ptr = filter_cpu;
    alloc_desc(filterTensorOrg, desc);

    tmpBytes = tensorNumBytes(inputDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);
    TensorDesc filterDescTran;
    std::vector<GCLMem_t> filterArray;
    std::vector<GCLMem_t> outputArray;
    std::vector<GCLMem_t> biasArray;
    filterArray.push_back(filter);
    outputArray.push_back(output);
    biasArray.push_back(bias);

    CHECK_STATUS(
        fully_connected_transform_filter(inputTensor, filterTensorOrg, &filterTensor, &archInfo));

    CHECK_STATUS(ocl_set_input(handle, input, inputDesc, input_cpu, tmpbuf, true));

    CHECK_STATUS(
        fully_connected(inputTensor, filterTensor, biasTensor, tmpTensor, outputTensor, &archInfo));
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
    outputDesc = outputTensor.get_desc();
    CHECK_STATUS(ocl_get_output(handle, output, outputDesc, true));
    output_gpu = output->mapPtrArray.back();

    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u)+(%u %u)=(%u %u %u %u)", in, ic, ih, iw, fn, fc, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "InnerProdect", params);
#ifdef _DEBUG
    double ops = 2.0 * ow * oh * fc + 1.0 * ow * oh;
    ut_log(dt, buffer, ops, time);
#endif
    if (row > 1) {
        filterDesc = tensor2df(dt, DF_TRANSPOSE, fn, fc);
    }
    Tensor inputTensorCpu;
    inputTensorCpu.resize(inputDesc);
    inputTensorCpu.alloc();
    memcpy(get_ptr_from_tensor(inputTensorCpu, UT_ARCH), input_cpu, tensorNumBytes(inputDesc));

    Tensor filterTensorCpu;
    filterTensorCpu.resize(filterDesc);
    filterTensorCpu.alloc();
    memcpy(get_ptr_from_tensor(filterTensorCpu, UT_ARCH), filter_cpu, tensorNumBytes(filterDesc));

    Tensor biasTensorCpu;
    biasTensorCpu.resize(biasDesc);
    biasTensorCpu.alloc();
    memcpy(get_ptr_from_tensor(biasTensorCpu, UT_ARCH), bias_cpu, tensorNumBytes(biasDesc));

    Tensor outputTensorCpu;
    outputTensorCpu.resize(outputDesc_cpu);
    outputTensorCpu.alloc();

    Tensor tmpTensorCpu;
    CHECK_STATUS(fully_connected_infer_forward_tmp_bytes(
        inputTensorCpu, filterTensorCpu, &tmpBytes, &archInfo_org));
    tmpTensorCpu.resize(tensor1d(DT_F16, tmpBytes / bytesOf(DT_F16)));
    tmpTensorCpu.alloc();

    CHECK_STATUS(fully_connected(inputTensorCpu, filterTensorCpu, biasTensorCpu, tmpTensorCpu,
        outputTensorCpu, &archInfo_org));
    ut_check_a(output_gpu, get_ptr_from_tensor(outputTensorCpu, UT_ARCH), on * oc * ow * oh, dt);

    free(input_cpu);
    free(filter_cpu);
    free(bias_cpu);
    return 0;
}
#endif

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    fullyConnectedTest(argc, argv, DT_F16);
#endif
    return 0;
}
