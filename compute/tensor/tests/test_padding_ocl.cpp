
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
int paddingTest(int argc, char **argv, DataType dt)
{
    // input dim
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 ih = atoi(argv[3]);
    U32 iw = atoi(argv[4]);

    // padding info
    U32 h_top = atoi(argv[5]);
    U32 h_bot = atoi(argv[7]);
    U32 w_lef = atoi(argv[6]);
    U32 w_rig = atoi(argv[8]);
    U32 mode = atoi(argv[9]);

    ArchInfo archInfo;
    archInfo.arch = MALI;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;
    PadParamSpec padParamSpec;

    padParamSpec.top = h_top;
    padParamSpec.bottom = h_bot;
    padParamSpec.left = w_lef;
    padParamSpec.right = w_rig;
    padParamSpec.constant_value = 0.0;
    switch (mode) {
        case 0: {
            padParamSpec.pad_mode = Pad_Constant;
            break;
        }
        case 1: {
            padParamSpec.pad_mode = Pad_Edge;
            break;
        }
        case 2: {
            // limitation: the h_fir and the h_sec should lower than 0
            padParamSpec.pad_mode = Pad_Reflect;
            break;
        }
        case 3: {
            padParamSpec.pad_mode = Pad_Symmetric;
            break;
        }
        default: {
            UNI_ERROR_LOG("unknown paddding mode %d\n", mode);
            break;
        }
    }

    TensorDesc inputDescCPU, inputDescGPU, outputDescCPU, outputDescGPU;
    inputDescCPU = tensor4df(dt, DF_NCHW, in, ic, ih, iw);
    inputDescGPU = tensor4df(dt, DF_NCHW, in, ic, ih, iw);

    U32 input_len = tensorNumElements(inputDescCPU);
    U8 *inputCPU = ut_input_v(input_len, dt, UT_INIT_RANDOM);
    U8 *outputGPU = NULL;
    F16 *val = (F16 *)inputCPU;
    for (U32 i = 0; i < input_len; i++)
        val[i] = i;

    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    Tensor inputTensor = Tensor(OCLMem);
    Tensor outputTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    inputTensor.resize(inputDescGPU);
    U32 str[3] = {1, 1, 1};
    U32 off[3] = {0, 0, 0};
    GCLMemDesc inputMemDesc = gcl_mem_desc(str, off, DT_U8, DF_NCWHC4);
    ocl_set_desc(&inputTensor, inputMemDesc);

    MaliPara maliPara;
    maliPara.handle = handle;
    archInfo.archPara = &maliPara;

    CHECK_STATUS(padding_infer_output_size(&inputTensor, padParamSpec, &outputTensor, &archInfo));

    GCLMem_t output = alloc_map(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));

    U32 maxBytes = 0;
    U32 tmpBytes = 0;
    tmpBytes = tensorNumBytes(inputDescGPU);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(ocl_set_input(handle, input, inputDescGPU, inputCPU, tmpbuf, true));
    CHECK_STATUS(padding(inputTensor, padParamSpec, outputTensor, &archInfo));
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
    outputDescGPU = outputTensor.get_desc();
    CHECK_STATUS(ocl_get_output(handle, output, outputDescGPU, true));
    outputGPU = output->mapPtrArray.back();

    char buffer[150];
    char params[120];
    U32 on, oc, oh, ow;
    on = outputDescGPU.dims[3];
    oc = outputDescGPU.dims[2];
    oh = outputDescGPU.dims[1];
    ow = outputDescGPU.dims[0];
    sprintf(params, "(%u %u %u %u)->(%u %u %u %u)", in, ic, ih, iw, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "padding", params);
#ifdef _DEBUG
    double ops = on * oc * oh * ow * 4;  // TO DO
    ut_log(dt, buffer, ops, time);
#endif
    Tensor inputTensorCpu;
    inputTensorCpu.resize(inputDescCPU);
    inputTensorCpu.alloc();
    memcpy(get_ptr_from_tensor(inputTensorCpu, UT_ARCH), inputCPU, tensorNumBytes(inputDescCPU));

    Tensor outputTensorCpu;
    CHECK_STATUS(
        padding_infer_output_size(&inputTensorCpu, padParamSpec, &outputTensorCpu, &archInfo_org));
    outputTensorCpu.alloc();

    if (UT_CHECK) {
        CHECK_STATUS(padding(inputTensorCpu, padParamSpec, outputTensorCpu, &archInfo_org));
    }
    TensorDesc desc = outputTensorCpu.get_desc();
    ut_check_a(
        outputGPU, get_ptr_from_tensor(outputTensorCpu, UT_ARCH), tensorNumElements(desc), dt);

    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(inputCPU);
    return 0;
}
#endif

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    paddingTest(argc, argv, DT_F16);
#endif
    return 0;
}
