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

void NCHWC8_to_NCHW(F16 *input_cpu, F16 *input_cpu_nchw, U32 ih, U32 iw, U32 ic)
{
    int index_c = 0;
    int index_hw = 0;
    int channel_k = 0;
    for (int i = 0; i < (int)(ic * ih * iw);) {
        index_c = i % (ih * iw);
        index_hw = i / (ih * iw);
        for (int k = 0; k < 8; k++) {
            if (index_hw % 8 == 0) {
                channel_k = index_hw * (ih * iw);
            }
            if (index_c == 0) {
                for (int j = 0; j < (int)(ih * iw); j++) {
                    input_cpu_nchw[i++] = input_cpu[channel_k + k + j * 8];
                }
            }
        }
    }
}

int poolingTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc == 18);
    // in data
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 it = atoi(argv[3]);
    U32 ih = atoi(argv[4]);
    U32 iw = atoi(argv[5]);

    PoolingParamSpec p;
    p.mode = POOLING_MEAN;
    p.round_mode = ROUND_CEIL;
    p.kernel_t = atoi(argv[6]);
    p.kernel_h = atoi(argv[7]);
    p.kernel_w = atoi(argv[8]);
    p.stride_t = atoi(argv[9]);
    p.stride_h = atoi(argv[10]);
    p.stride_w = atoi(argv[11]);
    p.pad_before = atoi(argv[12]);
    p.pad_after = atoi(argv[13]);
    p.pad_top = atoi(argv[14]);
    p.pad_bottom = atoi(argv[15]);
    p.pad_left = atoi(argv[16]);
    p.pad_right = atoi(argv[17]);

    ArchInfo archInfo;
    archInfo.arch = MALI;

    if (gcl_check_device_qualcomm(OCLContext::getInstance().handle.get())) {
        archInfo.arch = QUALCOMM;  //off qualcomm
    }

    TensorDesc inputDescCpu, inputDescGpu;
    if (it == 1) {
        inputDescCpu = tensor4df(dt, DF_NCHWC8, in, ic, ih, iw);
        inputDescGpu = tensor4df(dt, DF_NCHWC4, in, ic, ih, iw);
    } else {
        inputDescCpu = tensor5df(dt, DF_NCHWC8, in, ic, it, ih, iw);
        inputDescGpu = tensor5df(dt, DF_NCHWC4, in, ic, it, ih, iw);
    }
    TensorDesc output_desc_cpu, output_desc_gpu;
    U32 input_len = tensorNumElements(inputDescCpu);
    U8 *input_cpu_nchwc8 = ut_input_v(input_len, dt, UT_INIT_RANDOM);
    U8 *input_cpu_nchw = ut_input_v(input_len, dt, UT_INIT_ZERO);
    NCHWC8_to_NCHW((F16 *)input_cpu_nchwc8, (F16 *)input_cpu_nchw, ih, iw, ic);
    Tensor inputTensorCpu;
    inputTensorCpu.resize(inputDescCpu);
    inputTensorCpu.alloc();
    UNI_MEMCPY(get_ptr_from_tensor(inputTensorCpu, CPU_GENERAL), input_cpu_nchwc8,
        tensorNumBytes(inputDescCpu));

    Tensor outputTensorCpu;
    Tensor tmpTensorCpu;
    CHECK_STATUS(
        pooling_infer_output_size(&inputTensorCpu, p, &outputTensorCpu, &UT_SERIAL_ARCHINFO));

    outputTensorCpu.alloc();
    CHECK_STATUS(pooling(inputTensorCpu, p, tmpTensorCpu, outputTensorCpu, &UT_SERIAL_ARCHINFO));

    TensorDesc outputDescCpu = outputTensorCpu.get_desc();
    DataType odt;
    DataFormat odf;
    U32 on = 0, oc = 0, ot = 1, oh = 0, ow = 0;
    if (tensorIs4d(outputDescCpu)) {
        CHECK_STATUS(tensor4dGet(outputDescCpu, &odt, &odf, &on, &oc, &oh, &ow));
    } else if (tensorIs5d(outputDescCpu)) {
        CHECK_STATUS(tensor5dGet(outputDescCpu, &odt, &odf, &on, &oc, &ot, &oh, &ow));
    }

    U32 output_len = outputTensorCpu.length();
    U8 *output_cpu_nchw = ut_input_v(output_len, dt, UT_INIT_ZERO);
    NCHWC8_to_NCHW((F16 *)get_ptr_from_tensor(outputTensorCpu, CPU_GENERAL), (F16 *)output_cpu_nchw,
        oh, ow, oc);

    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    MemoryType memType = OCLMem;
    if (archInfo.arch == QUALCOMM) {
        memType = OCLMemImg;
    }
    Tensor inputTensor = Tensor(memType);
    Tensor outputTensor = Tensor(memType);
    Tensor tmpTensor = Tensor(OCLMem);
    inputTensor.resize(inputDescGpu);

    MaliPara maliPara;
    maliPara.handle = handle;
    archInfo.archPara = &maliPara;

    CHECK_STATUS(pooling_infer_output_size(&inputTensor, p, &outputTensor, &archInfo));
    TensorDesc outputDesc = outputTensor.get_desc();
    U8 *output_gpu = ut_input_v(on * oc * ot * oh * ow, dt, UT_INIT_ZERO);

    U32 maxBytes = 0;
    U32 tmpBytes = 0;
    CHECK_STATUS(pooling_infer_forward_tmp_bytes(inputTensor, outputTensor, &tmpBytes, &archInfo));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;

    GCLMem_t output = alloc(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));

    tmpBytes = tensorNumBytes(inputDescGpu);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(outputDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(ocl_set_input(handle, input, inputDescGpu, input_cpu_nchw, tmpbuf, true));
    CHECK_STATUS(pooling(inputTensor, p, tmpTensor, outputTensor, &archInfo));

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
    CHECK_STATUS(ocl_get_output(handle, output, outputDesc, output_gpu, tmpbuf, true));

    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u %u)/(%u %u %u)=(%u %u %u %u %u)", in, ic, it, ih, iw, p.kernel_t,
        p.kernel_h, p.kernel_w, on, oc, ot, oh, ow);
    sprintf(buffer, "%20s, %80s", "Pooling", params);
#ifdef _DEBUG
    double ops = 1.0 * output_len * p.kernel_t * p.kernel_h * p.kernel_w;
    ut_log(dt, buffer, ops, time);
#endif

    ut_check_a(output_gpu, output_cpu_nchw, on * oc * ot * ow * oh, dt);
    free(input_cpu_nchwc8);
    free(input_cpu_nchw);
    free(output_cpu_nchw);
    free(output_gpu);
    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    poolingTest(argc, argv, DT_F16);
#endif
    return 0;
}
