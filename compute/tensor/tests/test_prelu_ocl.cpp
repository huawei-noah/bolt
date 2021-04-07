
// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifdef _USE_FP16
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
inline GCLMem_t alloc_desc(Tensor tensor, GCLMemDesc desc)
{
    auto mem = (OclMemory *)tensor.get_memory();
    mem->padding(desc);
    mem->alloc();
    return (GCLMem_t)mem->get_ptr();
}

int preluTest(int argc, char **argv, DataType dt)
{
    // input dim
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 ih = atoi(argv[3]);
    U32 iw = atoi(argv[4]);
    U32 prop = atoi(argv[5]);
    U32 weightNum;

    ArchInfo archInfo;
    archInfo.arch = MALI;

    TensorDesc inputDescGPU, outputDescGPU, weightDescGPU;
    inputDescGPU = tensor4df(dt, DF_NCHW, in, ic, ih, iw);

    U32 input_len = tensorNumElements(inputDescGPU);
    U8 *inputCPU = ut_input_v(input_len, dt, UT_INIT_RANDOM);
    U8 *weightCPU = NULL;
    U8 *outputGPU = NULL;
    PReLUParamSpec preluDesc;
    if (prop) {
        preluDesc.propagate_down = true;
        weightCPU = ut_input_v(1, dt, UT_INIT_RANDOM);
        weightDescGPU = tensor1d(dt, 1);
    } else {
        preluDesc.propagate_down = false;
        weightCPU = ut_input_v(ic, dt, UT_INIT_RANDOM);
        weightDescGPU = tensor1d(dt, ic);
    }
    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    Tensor inputTensor = Tensor(OCLMem);
    Tensor outputTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    Tensor weightTensor = Tensor(OCLMem);
    inputTensor.resize(inputDescGPU);
    weightTensor.resize(weightDescGPU);

    MaliPara maliPara;
    maliPara.handle = handle;
    archInfo.archPara = &maliPara;

    CHECK_STATUS(prelu_infer_output_size(&inputTensor, &outputTensor, &archInfo));

    GCLMem_t output = alloc_map(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));

    GCLMemDesc desc = gclmem_build_desc();
    if (preluDesc.propagate_down) {
        weightNum = 1;
        desc.byteSize = weightNum * bytesOf(dt);
    } else {
        weightNum = (ic + 3) / 4 * 4;
        desc.byteSize = weightNum * bytesOf(dt);
    }
    desc.stride[0] = weightNum;
    desc.stride[1] = 1;
    desc.stride[2] = 1;
    desc.offset[0] = 0;
    desc.offset[1] = 0;
    desc.offset[2] = 0;
    desc.memType = GCL_MEM_BUF;
    desc.num = weightNum;
    desc.memFormat = DF_NHWC;
    desc.flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
    if (ic != 1) {
        U8 *weight_align = ut_input_v((ic + 3) / 4 * 4, dt, UT_INIT_ZERO);
        memcpy(weight_align, weightCPU, (ic + 3) / 4 * 4 * bytesOf(dt));
        desc.host_ptr = weight_align;
    } else {
        desc.host_ptr = weightCPU;
    }
    alloc_desc(weightTensor, desc);

    U32 tmpBytes;
    U32 maxBytes = 0;
    tmpBytes = tensorNumBytes(inputDescGPU);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(ocl_set_input(handle, input, inputDescGPU, inputCPU, tmpbuf, true));
    CHECK_STATUS(prelu(inputTensor, weightTensor, preluDesc, outputTensor, &archInfo));
    /*warp up*/
    UNI_INFO_LOG("warm up gpu:\n")
    for (U32 i = 0; i < 2; i++) {
        CHECK_STATUS(gcl_run_kernelVec(handle));
    }
    UNI_INFO_LOG("Run gpu:\n")
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernelVec_timing(handle, 0, handle->kernelVec->size()));
//    double time = handle->t_execute * 0.001;
#else
    CHECK_STATUS(gcl_run_kernelVec(handle));
#endif
    outputDescGPU = outputTensor.get_desc();
    CHECK_STATUS(ocl_get_output(handle, output, outputDescGPU, true));
    outputGPU = output->mapPtrArray.back();

    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u)->(%u %u %u %u)", in, ic, ih, iw, in, ic, ih, iw);
    sprintf(buffer, "%20s, %80s", "prelu", params);
    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(inputCPU);
    free(weightCPU);
    return 0;
}

int main(int argc, char **argv)
{
    preluTest(argc, argv, DT_F16);
    return 0;
}
#endif
