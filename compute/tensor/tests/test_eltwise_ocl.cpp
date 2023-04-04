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

int eltwiseTest(int argc, char *argv[], DataType dt)
{
    U32 num;
    U32 in, ic, ih, iw;
    U32 bn, bc, bh, bw;
    ArchInfo archInfo;
    archInfo.arch = MALI;

    num = 2;
    in = 1;
    ic = 4;
    ih = 4;
    iw = 4;
    bn = 1;
    bc = 4;
    bh = 4;
    bw = 4;

    if (argc >= 6) {
        num = atoi(argv[1]);
        in = atoi(argv[2]);
        ic = atoi(argv[3]);
        ih = atoi(argv[4]);
        iw = atoi(argv[5]);
        if (argc == 10) {
            bn = atoi(argv[6]);
            bc = atoi(argv[7]);
            bh = atoi(argv[8]);
            bw = atoi(argv[9]);
        } else {
            bn = in;
            bc = ic;
            bh = ih;
            bw = iw;
        }
    }

    EltwiseMode eltwiseMode = ELTWISE_SUM;
    EltwiseParamSpec eltwiseDesc;
    eltwiseDesc.mode = eltwiseMode;
    eltwiseDesc.activation_type = ACTIVATION_NULL;

    std::vector<void *> inputCpu(num);
    std::vector<Tensor> inTensorsCpu(num);
    std::vector<Tensor *> inTensorPtrCpu(num);
    TensorDesc inDesc = tensor4df(dt, DF_NCHW, in, ic, ih, iw);
    TensorDesc broadDesc = tensor4df(dt, DF_NCHW, bn, bc, bh, bw);
    Tensor outTensorCpu;
    for (U32 i = 0; i < num; i++) {
        if (i == 1) {
            inputCpu[i] = (void *)ut_input_v(bn * bc * bh * bw, dt, UT_INIT_RANDOM);
            inTensorsCpu[i].resize(broadDesc);
        } else {
            inputCpu[i] = (void *)ut_input_v(in * ic * ih * iw, dt, UT_INIT_RANDOM);
            inTensorsCpu[i].resize(inDesc);
        }
        inTensorsCpu[i].alloc();
        UNI_MEMCPY(get_ptr_from_tensor(inTensorsCpu[i], CPU_GENERAL), inputCpu[i],
            tensorNumBytes(inTensorsCpu[i].get_desc()));
        inTensorPtrCpu[i] = &inTensorsCpu[i];
    }
    CHECK_STATUS(
        eltwise_infer_output_size(inTensorPtrCpu, eltwiseDesc, &outTensorCpu, &UT_SERIAL_ARCHINFO));
    outTensorCpu.alloc();

    U32 maxBytes = 0;
    U32 tmpBytes;
    CHECK_STATUS(
        eltwise_infer_forward_tmp_bytes(inTensorsCpu, outTensorCpu, &tmpBytes, &UT_SERIAL_ARCHINFO));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    Tensor tmpTensorCpu;
    tmpTensorCpu.resize(tensor1d(DT_U8, tmpBytes));
    tmpTensorCpu.alloc();
    CHECK_STATUS(
        eltwise(inTensorsCpu, eltwiseDesc, tmpTensorCpu, outTensorCpu, &UT_SERIAL_ARCHINFO));

    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;

    std::vector<Tensor> inTensors(num);
    std::vector<Tensor *> inTensorPtr;
    Tensor outTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    inDesc.df = DF_NCHWC4;
    broadDesc.df = DF_NCHWC4;
    for (U32 i = 0; i < num; i++) {
        Tensor tensor = Tensor(OCLMem);
        if (i == 1) {
            tensor.resize(broadDesc);
        } else {
            tensor.resize(inDesc);
        }
        inTensors[i] = tensor;
        inTensorPtr.push_back(&inTensors[i]);
    }

    MaliPara maliPara;
    maliPara.handle = handle;
    archInfo.archPara = &maliPara;
    CHECK_STATUS(eltwise_infer_output_size(inTensorPtr, eltwiseDesc, &outTensor, &archInfo));

    CHECK_STATUS(eltwise_infer_forward_tmp_bytes(inTensors, outTensor, &tmpBytes, &archInfo));
    TensorDesc outputDesc = outTensor.get_desc();
    U8 *output_gpu = ut_input_v(tensorNumElements(outputDesc), dt, UT_INIT_RANDOM);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(inDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(broadDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(outputDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;

    std::vector<GCLMem_t> inputVec;
    GCLMem_t output = alloc(outTensor);
    for (U32 i = 0; i < num; i++) {
        GCLMem_t input = alloc(inTensors[i]);
        CHECK_STATUS(gcl_fill_memory_zero(handle, input));
        inputVec.push_back(input);
    }
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);
    for (U32 i = 0; i < num; i++) {
        TensorDesc desc = (i == 1) ? broadDesc : inDesc;
        CHECK_STATUS(ocl_set_input(handle, inputVec[i], desc, (U8 *)inputCpu[i], tmpbuf, true));
    }

    CHECK_STATUS(eltwise(inTensors, eltwiseDesc, tmpTensor, outTensor, &archInfo));

    for (U32 i = 0; i < UT_WARMUP; i++) {
        CHECK_STATUS(gcl_run_kernelVec(handle));
    }
    CHECK_STATUS(gcl_finish(handle));

    double time = 0;
#ifdef _DEBUG
    for (I32 i = 0; i < UT_LOOPS; i++) {
        CHECK_STATUS(gcl_run_kernelVec_timing(handle, 0, handle->kernelVec->size()));
        time += handle->t_execute * 0.001;
    }
#else
    double start = ut_time_ms();
    for (I32 i = 0; i < UT_LOOPS; i++) {
        CHECK_STATUS(gcl_run_kernelVec(handle));
        CHECK_STATUS(gcl_finish(handle));
    }
    double end = ut_time_ms();
    time = (end - start);
#endif
    time /= UT_LOOPS;

    CHECK_STATUS(ocl_get_output(handle, output, outputDesc, output_gpu, tmpbuf, true));
    char buffer[150];
    char params[120];
    U32 on, oc, oh, ow;
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);
    if (num == 2) {
        sprintf(params, "(%u %u %u)+(%u %u %u)=(%u %u %u)", ic, ih, iw, bc, bh, bw, oc, oh, ow);
    } else {
        sprintf(params, "%u (%u %u %u %u)=(%u %u %u %u)", num, in, ic, ih, iw, in, ic, ih, iw);
    }
    sprintf(buffer, "%20s, %80s", "eltwise", params);
    double ops = 1.9 * num * on * oc * oh * ow;
    ut_log(dt, buffer, ops, time);
    ut_check_v(output_gpu, get_ptr_from_tensor(outTensorCpu, CPU_GENERAL), on * oc * oh * ow, dt, 0.3);

    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    for (U32 i = 0; i < num; i++) {
        free(inputCpu[i]);
    }
    free(output_gpu);

    return 0;
}

int main(int argc, char **argv)
{
    eltwiseTest(argc, argv, DT_F16);
    return 0;
}
