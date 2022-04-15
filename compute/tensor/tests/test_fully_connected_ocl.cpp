// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "tensor_computing.h"
#include "ut_util_ocl.h"

int fullyConnectedTest(int argc, char *argv[], DataType dt)
{
    U32 in, ic, ih, iw;
    U32 fn, fc;
    U32 on, oc, oh, ow;
    U32 biasNum;
    ArchInfo archInfo;
    archInfo.arch = MALI;
    if (gcl_check_device_qualcomm(OCLContext::getInstance().handle.get())) {
        archInfo.arch = QUALCOMM;
    }

    in = 1;
    ic = 4;
    ih = 4;
    iw = 4;
    fc = 64;
    fn = 4;

    if (argc == 7) {
        in = atoi(argv[1]);
        ic = atoi(argv[2]);
        ih = atoi(argv[3]);
        iw = atoi(argv[4]);
        fc = atoi(argv[5]);
        fn = atoi(argv[6]);
    }

    if (iw != fc && in * ic * ih * iw != fc) {
        CHECK_STATUS(NOT_MATCH);
    }
    U32 row = (in * ic * ih * iw) / fc;

    TensorDesc inputDesc, filterDesc, outputDesc, biasDesc;
    TensorDesc filterDesc_cpu, outputDesc_cpu;

    inputDesc = tensor4df(dt, DF_NCHW, in, ic, ih, iw);
    filterDesc = tensor2df(dt, DF_NORMAL, fn, fc);
    outputDesc_cpu = tensor2df(dt, DF_NORMAL, oh, ow);
    biasDesc = tensor1d(dt, fn);

    U8 *input_cpu = ut_input_v(in * ic * ih * iw, dt, UT_INIT_RANDOM);
    U8 *filter_cpu = ut_input_v(fn * fc, dt, UT_INIT_RANDOM);
    U8 *bias_cpu = ut_input_v(fn, dt, UT_INIT_RANDOM);

    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    MemoryType memType = OCLMem;
    if (archInfo.arch == QUALCOMM && row > 1) {
        memType = OCLMemImg;
    }
    Tensor inputTensor = Tensor(memType);
    Tensor filterTensor = Tensor(memType);
    Tensor outputTensor = Tensor(OCLMem);
    Tensor filterTensorOrg = Tensor(OCLMem);
    Tensor biasTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    Tensor tmpTensorImg = Tensor(OCLMemImg);
    inputTensor.resize(inputDesc);
    filterTensorOrg.resize(filterDesc);
    biasTensor.resize(biasDesc);

    MaliPara maliPara;
    ForwardRunInfoMali runInfo;
    runInfo.algorithm = (I32)(CONVOLUTION_ALGORITHM_NULL);
    runInfo.best_h[0] = 1;
    runInfo.best_c[0] = 1;
    runInfo.best_k[0] = 1;
    maliPara.handle = handle;
    maliPara.forwardRunInfo = &runInfo;
    archInfo.archPara = &maliPara;

    CHECK_STATUS(
        fully_connected_infer_output_size(&inputTensor, filterTensorOrg, &outputTensor, &archInfo));
    CHECK_STATUS(fully_connected_infer_forward_algorithm(
        inputTensor, filterTensorOrg, outputTensor, &archInfo));

    U32 maxBytes[4] = {0};
    CHECK_STATUS(fully_connected_infer_forward_tmp_bytes(
        inputTensor, filterTensorOrg, outputTensor, maxBytes, &archInfo));

    TensorDesc ftmDesc;
    CHECK_STATUS(fully_connected_transform_filter_bytes(filterTensorOrg, &ftmDesc, &archInfo));

    GCLMem_t output = alloc(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    alloc_host_ptr(filterTensorOrg, filter_cpu);
    filterTensor.resize(ftmDesc);
    alloc(filterTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));

    U32 item_m = runInfo.best_h[0];
    biasNum = (fn + item_m - 1) / item_m * item_m;
    if (biasNum > fn) {
        U8 *bias_val = ut_input_v(biasNum, dt, UT_INIT_ZERO);
        UNI_MEMCPY(bias_val, bias_cpu, fn * bytesOf(dt));
        free(bias_cpu);
        bias_cpu = bias_val;
    }
    OclMemory *biasMem = (OclMemory *)biasTensor.get_memory();
    U32 pr = biasNum - fn;
    biasMem->padding(0, pr, 0, 0);
    alloc_host_ptr(biasTensor, bias_cpu);

    outputDesc = outputTensor.get_desc();
    U8 *output_gpu = ut_input_v(tensorNumElements(outputDesc), dt, UT_INIT_RANDOM);
    U32 tmpBytes;
    tmpBytes = tensorNumBytes(inputDesc);
    maxBytes[0] = (tmpBytes > maxBytes[0]) ? tmpBytes : maxBytes[0];
    tmpBytes = tensorNumBytes(outputDesc);
    maxBytes[0] = (tmpBytes > maxBytes[0]) ? tmpBytes : maxBytes[0];
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes[0]);
    alloc_img(tmpTensorImg, maxBytes + 1);
    std::vector<Tensor> tmpTensors(2);
    tmpTensors[0] = tmpTensor;
    tmpTensors[1] = tmpTensorImg;

    CHECK_STATUS(
        fully_connected_transform_filter(inputTensor, filterTensorOrg, &filterTensor, &archInfo));

    CHECK_STATUS(ocl_set_input(handle, input, inputDesc, input_cpu, tmpbuf, true));

    CHECK_STATUS(
        fully_connected(inputTensor, filterTensor, biasTensor, tmpTensors, outputTensor, &archInfo));

    /*warp up*/
    for (U32 i = 0; i < 2; i++) {
        CHECK_STATUS(gcl_run_kernelVec(handle));
    }

#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernelVec_timing(handle, 0, handle->kernelVec->size()));
    double time = handle->t_execute * 0.001;
#else
    CHECK_STATUS(gcl_run_kernelVec(handle));
#endif
    CHECK_STATUS(ocl_get_output(handle, output, outputDesc, output_gpu, tmpbuf, true));

    char buffer[150];
    char params[120];
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);
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
    UNI_MEMCPY(
        get_ptr_from_tensor(inputTensorCpu, CPU_GENERAL), input_cpu, tensorNumBytes(inputDesc));

    Tensor filterTensorCpu;
    filterTensorCpu.resize(filterDesc);
    filterTensorCpu.alloc();
    UNI_MEMCPY(
        get_ptr_from_tensor(filterTensorCpu, CPU_GENERAL), filter_cpu, tensorNumBytes(filterDesc));

    Tensor biasTensorCpu;
    biasTensorCpu.resize(biasDesc);
    biasTensorCpu.alloc();
    UNI_MEMCPY(get_ptr_from_tensor(biasTensorCpu, CPU_GENERAL), bias_cpu, tensorNumBytes(biasDesc));

    Tensor outputTensorCpu;
    outputTensorCpu.resize(outputDesc_cpu);
    outputTensorCpu.alloc();

    Tensor tmpTensorCpu;
    CHECK_STATUS(fully_connected_infer_forward_tmp_bytes(
        inputTensorCpu, filterTensorCpu, outputTensorCpu, &tmpBytes, &UT_SERIAL_ARCHINFO));
    tmpTensorCpu.resize(tensor1d(DT_F16, tmpBytes / bytesOf(DT_F16)));
    tmpTensorCpu.alloc();
    std::vector<Tensor> tmpTensorsCpu(1, tmpTensorCpu);

    CHECK_STATUS(fully_connected(inputTensorCpu, filterTensorCpu, biasTensorCpu, tmpTensorsCpu,
        outputTensorCpu, &UT_SERIAL_ARCHINFO));
    ut_check_a(output_gpu, get_ptr_from_tensor(outputTensorCpu, CPU_GENERAL), on * oc * ow * oh, dt);

    free(input_cpu);
    free(filter_cpu);
    free(bias_cpu);
    free(output_gpu);
    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    fullyConnectedTest(argc, argv, DT_F16);
#endif
    return 0;
}
