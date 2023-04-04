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

inline void topk_cpu_max(F16 *input, U32 len, U32 topk, F16 *output, I32 *outputId)
{
    F32 min_val = -UNI_F16_MAX;
    for (U32 i = 0; i < topk; i++) {
        U32 index = 0;
        F32 max_val = min_val;
        for (U32 j = 0; j < len; j++) {
            F32 val;
            transformToFloat(DT_F16, input + j, &val, 1);
            if (val > max_val) {
                max_val = val;
                index = j;
            }
        }
        transformFromFloat(DT_F16, &min_val, input + index, 1);
        transformFromFloat(DT_F16, &max_val, output + i, 1);
        outputId[i] = index;
    }
}

inline void sort_gpu_result(
    F16 *res_gpu, I32 *res_id_gpu, U32 topk, F16 *res_gpu_sort, I32 *res_id_gpu_sort)
{
    std::vector<U32> skip_j;
    for (U32 i = 0; i < topk; i++) {
        F32 max_val = -UNI_F16_MAX;
        I32 index = UNI_F16_MAX;
        U32 sj = 0;
        for (U32 j = 0; j < topk; j++) {
            bool skip = false;
            for (auto p : skip_j) {
                if (j == p) {
                    skip = true;
                }
            }
            if (!skip) {
                F32 val;
                transformToFloat(DT_F16, res_gpu + j, &val, 1);
                if (val > max_val) {
                    max_val = val;
                    index = res_id_gpu[j];
                    sj = j;
                } else if (val == max_val) {
                    if (res_id_gpu[j] < index) {
                        index = res_id_gpu[j];
                        sj = j;
                    }
                }
            }
        }
        transformFromFloat(DT_F16, &max_val, res_gpu_sort + i, 1);
        res_id_gpu_sort[i] = index;
        skip_j.push_back(sj);
    }
}

int topkTest(int argc, char **argv, DataType dt)
{
    U32 in = 1;
    U32 ic = 1;
    U32 ih = 1;
    U32 iw = 3000;
    TopKParamSpec p;
    p.axis = 0;
    p.k = 30;
    p.largest = 1;
    p.sorted = 0;
    if (argc == 8) {
        in = atoi(argv[1]);
        ic = atoi(argv[2]);
        ih = atoi(argv[3]);
        iw = atoi(argv[4]);
        p.axis = atof(argv[5]);
        p.k = atof(argv[6]);
        p.largest = atof(argv[7]);
        p.sorted = atof(argv[8]);
    }
    U32 on, oc, oh, ow;

    ArchInfo archInfo;
    archInfo.arch = MALI;

    U32 len = in * ic * ih * iw;

    TensorDesc input_desc_cpu = tensor1d(dt, len);
    TensorDesc output_desc_cpu = tensor1d(dt, (U32)p.k);
    TensorDesc output_indices_desc_cpu = tensor1d(DT_I32, (U32)p.k);
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
    output_desc_gpu = outputTensor.get_desc();
    output_indices_desc_gpu = outputIndicesTensor.get_desc();
    output_gpu = ut_input_v(tensorNumElements(output_desc_gpu), dt, UT_INIT_RANDOM);
    output_indices_gpu =
        ut_input_v(tensorNumElements(output_indices_desc_gpu), DT_I32, UT_INIT_RANDOM);

    GCLMem_t output = alloc(outputTensor);
    GCLMem_t outputIndices = alloc(outputIndicesTensor);
    GCLMem_t input = alloc(inputTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));

    U32 maxBytes = 0;
    U32 tmpBytes = 0;
    tmpBytes = tensorNumBytes(input_desc_gpu);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(output_desc_gpu);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(output_indices_desc_gpu);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    CHECK_STATUS(topk_infer_forward_tmp_bytes(inputTensor, p, outputTensor, &tmpBytes, &archInfo));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(ocl_set_input(handle, input, input_desc_gpu, input_cpu, tmpbuf, true));
    CHECK_STATUS(topk(inputTensor, p, tmpTensor, outputTensor, outputIndicesTensor, &archInfo));

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

    CHECK_STATUS(ocl_get_output(handle, output, output_desc_gpu, output_gpu, tmpbuf, true));
    CHECK_STATUS(ocl_get_output(
        handle, outputIndices, output_indices_desc_gpu, output_indices_gpu, tmpbuf, true));
    tensorSelectGet(output_desc_gpu, NULL, NULL, &on, &oc, &oh, &ow);
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u) = (%u %u %u %u)", in, ic, ih, iw, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "Topk", params);
    double ops = tensorNumElements(output_desc_gpu);
    ut_log(dt, buffer, ops, time);

    F16 *output_cpu = (F16 *)malloc(sizeof(F16) * p.k);
    I32 *output_id_cpu = (I32 *)malloc(sizeof(I32) * p.k);
    F16 *res_gpu_sort = (F16 *)malloc(sizeof(F16) * p.k);
    I32 *res_id_gpu_sort = (I32 *)malloc(sizeof(I32) * p.k);
    topk_cpu_max((F16 *)input_cpu, len, p.k, output_cpu, output_id_cpu);
    sort_gpu_result(
        (F16 *)output_gpu, (I32 *)output_indices_gpu, p.k, res_gpu_sort, res_id_gpu_sort);

    ut_check_v(res_gpu_sort, output_cpu, p.k, dt, 0.1);
    ut_check_v(res_id_gpu_sort, output_id_cpu, p.k, dt, 0.1);

    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(output_cpu);
    free(output_id_cpu);
    free(res_gpu_sort);
    free(res_id_gpu_sort);
    free(input_cpu);
    free(output_gpu);
    free(output_indices_gpu);
    return 0;
}

int main(int argc, char **argv)
{
    topkTest(argc, argv, DT_F16);
    return 0;
}
