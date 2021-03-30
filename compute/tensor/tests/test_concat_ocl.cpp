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
#include "ut_util.h"
#include "libkernelsource.h"
#include <string.h>
#include "gcl.h"
#include <iostream>

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

int concatTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc > 2);
    ConcatParamSpec p;
    int num = atoi(argv[1]);
    p.axis = atoi(argv[2]);
    CHECK_REQUIREMENT(argc == 1 + 2 + (num + 1) * 4);
    std::vector<TensorDesc> inputDesc(num);
    std::vector<Tensor> inputTensorCpu;
    std::vector<Tensor> inputTensor;
    for (int i = 0; i < num; i++) {
        U32 n, c, h, w;
        n = atoi(argv[3 + i * 4]);
        c = atoi(argv[3 + i * 4 + 1]);
        h = atoi(argv[3 + i * 4 + 2]);
        w = atoi(argv[3 + i * 4 + 3]);
        inputDesc[i] = tensor4df(dt, DF_NCHW, n, c, h, w);
        std::shared_ptr<Tensor> tensorCpu(new Tensor());
        std::shared_ptr<Tensor> tensor(new Tensor(OCLMem));
        tensorCpu->resize(inputDesc[i]);
        tensor->resize(inputDesc[i]);
        U32 str[3] = {1, 1, 1};
        U32 off[3] = {0, 0, 0};
        GCLMemDesc inputMemDesc = gcl_mem_desc(str, off, DT_U8, DF_NCWHC4);
        ocl_set_desc(tensor.get(), inputMemDesc);
        inputTensorCpu.push_back(*tensorCpu.get());
        inputTensor.push_back(*tensor.get());
    }
    U32 on = atoi(argv[3 + num * 4]);
    U32 oc = atoi(argv[3 + num * 4 + 1]);
    U32 oh = atoi(argv[3 + num * 4 + 2]);
    U32 ow = atoi(argv[3 + num * 4 + 3]);

    ArchInfo archInfo;
    archInfo.arch = MALI;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;

    std::vector<Tensor *> inputTensorCpuPtr;
    std::vector<Tensor *> inputTensorPtr;
    for (int i = 0; i < num; i++) {
        inputTensorCpuPtr.push_back(&inputTensorCpu[i]);
    }
    for (int i = 0; i < num; i++) {
        inputTensorPtr.push_back(&inputTensor[i]);
    }

    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    Tensor outputTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);

    U32 in_len = 0;
    for (int i = 0; i < num; i++) {
        in_len += tensorNumElements(inputDesc[i]);
    }
    std::vector<void *> input_cpu(num);
    U8 *tmp = ut_input_v(in_len, dt, UT_INIT_RANDOM);
    U32 count = 0;
    for (int i = 0; i < num; i++) {
        input_cpu[i] = (void *)(tmp + count * bytesOf(dt));
        count += tensorNumElements(inputDesc[i]);
    }

    MaliPara maliPara;
    maliPara.handle = handle;
    archInfo.archPara = &maliPara;

    CHECK_STATUS(concat_infer_output_size(inputTensorPtr, p, &outputTensor, &archInfo));
    TensorDesc outputDesc = outputTensor.get_desc();

    U32 maxBytes = 0;
    U32 tmpBytes = 0;
    CHECK_STATUS(concat_infer_forward_tmp_bytes(inputTensor, &tmpBytes, &archInfo));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;

    GCLMem_t output = alloc_map(outputTensor);
    for (int i = 0; i < num; i++) {
        tmpBytes = tensorNumBytes(inputTensor[i].get_desc());
        maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    }
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    for (int i = 0; i < num; i++) {
        GCLMem_t input = alloc(inputTensor[i]);
        CHECK_STATUS(gcl_fill_memory_zero(handle, input));
        CHECK_STATUS(ocl_set_input(handle, input, inputDesc[i], (U8 *)input_cpu[i], tmpbuf, true));
    }

    CHECK_STATUS(concat(inputTensor, p, tmpTensor, outputTensor, &archInfo));

    /*warp up*/
    UNI_INFO_LOG("warm up gpu:\n")
    for (U32 i = 0; i < 2; i++) {
        CHECK_STATUS(gcl_run_kernelVec(handle));
    }

    UNI_INFO_LOG("Run:\n")
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernelVec_timing(handle, 0, handle->kernelVec->size()));
    double time = handle->t_execute * 0.001;
#else
    CHECK_STATUS(gcl_run_kernelVec(handle));
#endif
    CHECK_STATUS(ocl_get_output(handle, output, outputDesc, true));
    U8 *output_gpu_val = output->mapPtrArray.back();

    char buffer[150];
    char params[120];
    sprintf(params, "%d (*)/%u=(%u %u %u %u)", num, p.axis, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "Concat", params);
#ifdef _DEBUG
    double ops = 1.0 * on * oc * oh * ow;
    ut_log(dt, buffer, ops, time);
#endif
    for (int i = 0; i < num; i++) {
        inputTensorCpu[i].alloc();
        memcpy(get_ptr_from_tensor(inputTensorCpu[i], UT_ARCH), input_cpu[i],
            tensorNumBytes(inputDesc[i]));
    }

    Tensor outputTensorCpu;
    CHECK_STATUS(concat_infer_output_size(inputTensorCpuPtr, p, &outputTensorCpu, &archInfo_org));
    outputTensorCpu.alloc();

    Tensor tmpTensorCpu;
    CHECK_STATUS(concat(inputTensorCpu, p, tmpTensorCpu, outputTensorCpu, &archInfo_org));
    ut_check_a(output_gpu_val, get_ptr_from_tensor(outputTensorCpu, UT_ARCH), on * oc * ow * oh, dt);

    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(tmp);
    return 0;
}
#endif
int main(int argc, char **argv)
{
#ifdef _USE_FP16
    concatTest(argc, argv, DT_F16);
#endif
    return 0;
}
