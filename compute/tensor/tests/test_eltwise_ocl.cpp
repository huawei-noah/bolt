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
int eltwiseTest(int argc, char *argv[], DataType dt)
{
    U32 num;
    U32 in, ic, ih, iw;
    U32 bn, bc, bh, bw;
    ArchInfo archInfo;
    archInfo.arch = MALI;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;

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
    eltwiseDesc.elt_mode = eltwiseMode;
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
            F16 *val = (F16 *)inputCpu[i];
            inTensorsCpu[i].resize(broadDesc);
        } else {
            inputCpu[i] = (void *)ut_input_v(in * ic * ih * iw, dt, UT_INIT_RANDOM);
            F16 *val = (F16 *)inputCpu[i];
            inTensorsCpu[i].resize(inDesc);
        }
        inTensorsCpu[i].alloc();
        memcpy(get_ptr_from_tensor(inTensorsCpu[i], UT_ARCH), inputCpu[i],
            tensorNumBytes(inTensorsCpu[i].get_desc()));
        inTensorPtrCpu[i] = &inTensorsCpu[i];
    }
    CHECK_STATUS(eltwise_infer_output_size(inTensorPtrCpu, &outTensorCpu, &archInfo_org));
    outTensorCpu.alloc();

    U32 maxBytes = 0;
    U32 tmpBytes;
    CHECK_STATUS(
        eltwise_infer_forward_tmp_bytes(inTensorsCpu, outTensorCpu, &tmpBytes, &archInfo_org));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    Tensor tmpTensorCpu;
    tmpTensorCpu.resize(tensor1d(DT_U8, tmpBytes));
    tmpTensorCpu.alloc();
    CHECK_STATUS(eltwise(inTensorsCpu, eltwiseDesc, tmpTensorCpu, outTensorCpu, &archInfo_org));

    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;

    std::vector<Tensor> inTensors(num);
    std::vector<Tensor *> inTensorPtr;
    Tensor outTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    for (U32 i = 0; i < num; i++) {
        Tensor tensor = Tensor(OCLMem);
        U32 str[3] = {1, 1, 1};
        U32 off[3] = {0, 0, 0};
        GCLMemDesc inputMemDesc = gcl_mem_desc(str, off, DT_U8, DF_NCWHC4);
        if (i == 1) {
            tensor.resize(broadDesc);
            inputMemDesc = gcl_mem_desc(str, off, DT_U8, DF_NCHW);
        } else {
            tensor.resize(inDesc);
        }
        ocl_set_desc(&tensor, inputMemDesc);
        inTensors[i] = tensor;
        inTensorPtr.push_back(&inTensors[i]);
    }

    MaliPara maliPara;
    maliPara.handle = handle;
    archInfo.archPara = &maliPara;
    CHECK_STATUS(eltwise_infer_output_size(inTensorPtr, &outTensor, &archInfo));

    CHECK_STATUS(eltwise_infer_forward_tmp_bytes(inTensors, outTensor, &tmpBytes, &archInfo));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(inDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(broadDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;

    std::vector<GCLMem_t> inputVec;
    GCLMem_t output = alloc_map(outTensor);
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

    /*warp up*/
    UNI_INFO_LOG("warm up gpu:\n")
    for (U32 i = 0; i < 2; i++) {
        CHECK_STATUS(gcl_run_kernelVec(handle));
    }

#ifdef _DEBUG
    //    std::vector<U32> kernelIndex;
    //    for (U32 i = 0; i < handle->kernelVec->size(); i++) {
    //        kernelIndex.push_back(i);
    //    }
    //    CHECK_STATUS(gcl_run_kernelVec_select_ls(handle, kernelIndex));
    CHECK_STATUS(gcl_finish(handle));
    double time = 0;
    double min_time = DBL_MAX;
    double max_time = 0;
    U32 loop = 1;
    for (U32 i = 0; i < loop; i++) {
        CHECK_STATUS(gcl_run_kernelVec_timing(handle, 0, handle->kernelVec->size()));
        double t = handle->t_execute * 0.001;
        if (t < min_time)
            min_time = t;
        if (t > max_time)
            max_time = t;
        time += t;
    }
    time = (time - min_time - max_time) / (loop - 2);
    UNI_INFO_LOG("min_time = %lf\n", min_time);
    UNI_INFO_LOG("max_time = %lf\n", max_time);
    UNI_INFO_LOG("avg_time = %lf\n", time);
    time = min_time;
#else
    CHECK_STATUS(gcl_run_kernelVec(handle));
#endif
    TensorDesc outputDesc = outTensor.get_desc();
    CHECK_STATUS(ocl_get_output(handle, output, outputDesc, true));
    U8 *output_gpu = output->mapPtrArray.back();
    char buffer[150];
    char params[120];
    if (num == 2) {
        sprintf(params, "(%u %u %u)+(%u %u %u)=(%u %u %u)", ic, ih, iw, bc, bh, bw, ic, ih, iw);
    } else {
        sprintf(params, "%u (%u %u %u %u)=(%u %u %u %u)", num, in, ic, ih, iw, in, ic, ih, iw);
    }
    sprintf(buffer, "%20s, %80s", "eltwise", params);
#ifdef _DEBUG
    double ops = 1.9 * num * in * ic * ih * iw;
    ut_log(dt, buffer, ops, time);
#endif
    ut_check_a(output_gpu, get_ptr_from_tensor(outTensorCpu, UT_ARCH), ic * ih * iw, dt);

    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    for (U32 i = 0; i < num; i++) {
        free(inputCpu[i]);
    }

    return 0;
}
#endif

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    eltwiseTest(argc, argv, DT_F16);
#endif
    return 0;
}
