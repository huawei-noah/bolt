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

inline void topk_cpu_max(F16 *input, U32 len, U32 topk, F16 *output, I32 *outputId)
{
    for (U32 i = 0; i < topk; i++) {
        U32 index = 0;
        F16 max_val = -65536;
        for (U32 j = 0; j < len; j++) {
            if (input[j] > max_val) {
                max_val = input[j];
                index = j;
            }
        }
        input[index] = -65536;
        output[i] = max_val;
        outputId[i] = index;
    }
}

inline void sort_gpu_result(
    F16 *res_gpu, I32 *res_id_gpu, U32 topk, F16 *res_gpu_sort, I32 *res_id_gpu_sort)
{
    std::vector<U32> skip_j;
    for (U32 i = 0; i < topk; i++) {
        F16 max_val = -65536;
        I32 index = 65536;
        U32 sj = 0;
        for (U32 j = 0; j < topk; j++) {
            bool skip = false;
            for (auto p : skip_j) {
                if (j == p) {
                    skip = true;
                }
            }
            if (!skip) {
                if (res_gpu[j] > max_val) {
                    max_val = res_gpu[j];
                    index = res_id_gpu[j];
                    sj = j;
                } else if (res_gpu[j] == max_val) {
                    if (res_id_gpu[j] < index) {
                        index = res_id_gpu[j];
                        sj = j;
                    }
                }
            }
        }
        res_gpu_sort[i] = max_val;
        res_id_gpu_sort[i] = index;
        skip_j.push_back(sj);
    }
}

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

int topkTest(int argc, char **argv, DataType dt)
{
    U32 in = 1;
    U32 ic = 1;
    U32 ih = 1;
    U32 iw = 3000;
    TopKParamSpec p;
    p.axis = 0;
    p.topk = 30;
    p.largest = 1;
    p.sorted = 0;
    if (argc == 8) {
        in = atoi(argv[1]);
        ic = atoi(argv[2]);
        ih = atoi(argv[3]);
        iw = atoi(argv[4]);
        p.axis = atof(argv[5]);
        p.topk = atof(argv[6]);
        p.largest = atof(argv[7]);
        p.sorted = atof(argv[8]);
    }
    U32 on, oc, oh, ow;

    ArchInfo archInfo;
    archInfo.arch = MALI;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;
    U32 len = in * ic * ih * iw;

    TensorDesc input_desc_cpu = tensor1d(dt, len);
    TensorDesc output_desc_cpu = tensor1d(dt, (U32)p.topk);
    TensorDesc output_indices_desc_cpu = tensor1d(DT_I32, (U32)p.topk);
    TensorDesc input_desc_gpu = tensor1d(dt, len);
    TensorDesc output_desc_gpu, output_indices_desc_gpu;

    U8 *input_cpu = ut_input_v(len, dt, UT_INIT_RANDOM);
    U8 *output_gpu = NULL;
    U8 *output_indices_gpu = NULL;

    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    Tensor inputTensor = Tensor(OCLMem);
    Tensor outputTensor = Tensor(OCLMem);
    Tensor outputIndicesTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);

    inputTensor.resize(input_desc_gpu);
    MaliPara maliPara;
    maliPara.handle = handle;
    archInfo.archPara = &maliPara;
    CHECK_STATUS(
        topk_infer_output_size(&inputTensor, p, &outputTensor, &outputIndicesTensor, &archInfo));

    GCLMem_t output = alloc_map(outputTensor);
    GCLMem_t outputIndices = alloc_map(outputIndicesTensor);
    GCLMem_t input = alloc(inputTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));

    U32 maxBytes = 0;
    U32 tmpBytes = 0;
    tmpBytes = tensorNumBytes(input_desc_gpu);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    CHECK_STATUS(topk_infer_forward_tmp_bytes(inputTensor, p, outputTensor, &tmpBytes, &archInfo));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(ocl_set_input(handle, input, input_desc_gpu, input_cpu, tmpbuf, true));
    CHECK_STATUS(topk(inputTensor, p, tmpTensor, outputTensor, outputIndicesTensor, &archInfo));
    /*warp up*/
    UNI_INFO_LOG("Warp up gpu:\n")
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

    output_desc_gpu = outputTensor.get_desc();
    output_indices_desc_gpu = outputIndicesTensor.get_desc();
    CHECK_STATUS(ocl_get_output(handle, output, output_desc_gpu, true));
    CHECK_STATUS(ocl_get_output(handle, outputIndices, output_indices_desc_gpu, true));
    output_gpu = output->mapPtrArray.back();
    output_indices_gpu = outputIndices->mapPtrArray.back();
    tensorSelectGet(output_desc_gpu, NULL, NULL, &on, &oc, &oh, &ow);
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u) = (%u %u %u %u)", in, ic, ih, iw, on, oc, oh, ow);
    sprintf(buffer, "16bit%20s, %80s", "topk", params);
    std::cout << buffer << std::endl;

    F16 *output_cpu = (F16 *)malloc(sizeof(F16) * p.topk);
    I32 *output_id_cpu = (I32 *)malloc(sizeof(I32) * p.topk);
    F16 *res_gpu_sort = (F16 *)malloc(sizeof(F16) * p.topk);
    I32 *res_id_gpu_sort = (I32 *)malloc(sizeof(I32) * p.topk);
    topk_cpu_max((F16 *)input_cpu, len, p.topk, output_cpu, output_id_cpu);
    sort_gpu_result(
        (F16 *)output_gpu, (I32 *)output_indices_gpu, p.topk, res_gpu_sort, res_id_gpu_sort);

    ut_check_a(res_gpu_sort, output_cpu, p.topk, dt);
    ut_check_a(res_id_gpu_sort, output_id_cpu, p.topk, dt);

    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(output_cpu);
    free(output_id_cpu);
    free(res_gpu_sort);
    free(res_id_gpu_sort);
    free(input_cpu);
    return 0;
}

int main(int argc, char **argv)
{
    topkTest(argc, argv, DT_F16);
    return 0;
}
#endif
