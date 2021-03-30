// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <vector>
#include <string.h>
#include "tensor_computing.h"
#include "ut_util.h"
#include "gcl.h"
#include "libkernelsource.h"
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

int transposeTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc == 9 || argc == 11);
    U32 in = 1;
    U32 ic = 1;
    U32 ih = 1;
    U32 iw = 1;
    U32 it = 1;
    TransposeParamSpec p, p_inv;
    TensorDesc inDesc;
    DataFormat df = DF_NCHW;
    TensorDesc inputDesc_cpu, inputDesc_gpu;

    if (argc == 9) {
        in = atoi(argv[1]);
        ic = atoi(argv[2]);
        ih = atoi(argv[3]);
        iw = atoi(argv[4]);
        p.trans_size = 4;
        p_inv.trans_size = 4;
        for (int i = 0; i < 4; i++) {
            I32 value = atoi(argv[5 + i]);
            p.trans_dims[i] = value;
            p_inv.trans_dims[value] = i;
        }
        inputDesc_cpu = tensor4df(dt, DF_NCHW, in, ic, ih, iw);
        inputDesc_gpu = tensor4df(dt, DF_NCHW, in, ic, ih, iw);
    } else {
        in = atoi(argv[1]);
        ic = atoi(argv[2]);
        it = atoi(argv[3]);
        ih = atoi(argv[4]);
        iw = atoi(argv[5]);
        p.trans_size = 5;
        p_inv.trans_size = 5;
        for (int i = 0; i < 5; i++) {
            I32 value = atoi(argv[6 + i]);
            p.trans_dims[i] = value;
            p_inv.trans_dims[value] = i;
        }
        inputDesc_cpu = tensor5df(dt, DF_NCHW, in, ic, it, ih, iw);
        inputDesc_gpu = tensor5df(dt, DF_NCHW, in, ic, it, ih, iw);
    }

    ArchInfo archInfo;
    ArchInfo archInfo_org;
    archInfo.arch = MALI;
    archInfo_org.arch = CPU_GENERAL;

    TensorDesc outputDesc;
    U32 len = tensorNumElements(inputDesc_cpu);
    U8 *input_cpu = ut_input_v(len, dt, UT_INIT_RANDOM);

    Tensor inputTensorCpu;
    inputTensorCpu.resize(inputDesc_cpu);
    inputTensorCpu.alloc();
    memcpy(get_ptr_from_tensor(inputTensorCpu, UT_ARCH), input_cpu, tensorNumBytes(inputDesc_cpu));
    Tensor outputTensorCpu;
    Tensor tmpTensorCpu;
    //run on cpu
    CHECK_STATUS(transpose_infer_output_size(&inputTensorCpu, p, &outputTensorCpu, &archInfo_org));
    outputTensorCpu.alloc();
    CHECK_STATUS(transpose(inputTensorCpu, p, tmpTensorCpu, outputTensorCpu, &archInfo_org));
    //run on gpu
    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    Tensor inputTensor = Tensor(OCLMem);
    Tensor outputTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    inputTensor.resize(inputDesc_gpu);
    U8 *output_gpu = NULL;
    MaliPara maliPara;
    maliPara.handle = handle;
    archInfo.archPara = &maliPara;

    CHECK_STATUS(transpose_infer_output_size(&inputTensor, p, &outputTensor, &archInfo));

    U32 maxBytes = 0;
    U32 tmpBytes;
    CHECK_STATUS(transpose_infer_forward_tmp_bytes(inputTensor, outputTensor, &tmpBytes, &archInfo))
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;

    GCLMem_t output = alloc_map(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));

    tmpBytes = tensorNumBytes(inputDesc_gpu);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(ocl_set_input(handle, input, inputDesc_gpu, input_cpu, tmpbuf, true));
    CHECK_STATUS(transpose(inputTensor, p, tmpTensor, outputTensor, &archInfo));
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
    outputDesc = outputTensor.get_desc();
    CHECK_STATUS(ocl_get_output(handle, output, outputDesc, true));
    output_gpu = output->mapPtrArray.back();

    char buffer[150];
    char params[120];
    U32 on, oc, ot, oh, ow;
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow, &ot);
    sprintf(params, "(%u %u %u %u %u)=(%u %u %u %u %u)", in, ic, it, ih, iw, on, oc, ot, oh, ow);
    sprintf(buffer, "%20s, %80s", "Transpose", params);
#ifdef _DEBUG
    double ops = len;
    ut_log(dt, buffer, ops, time);
#endif
    ut_check_a(output_gpu, get_ptr_from_tensor(outputTensorCpu, UT_ARCH), len, dt);

    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(input_cpu);
    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    transposeTest(argc, argv, DT_F16);
#endif
    return 0;
}
