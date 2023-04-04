
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

inline TensorDesc createDesc(U32 nDims, DataType dt, U32 n, U32 c, U32 h, U32 w)
{
    TensorDesc desc;
    desc.dt = dt;
    desc.nDims = nDims;
    desc.dims[0] = w;
    desc.dims[1] = h;
    desc.dims[2] = c;
    desc.dims[3] = n;
    if (nDims > 4) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (nDims == 4) {
        desc.df = DF_NCHW;
    } else if (nDims == 3) {
        desc.df = DF_MTK;
    } else {
        desc.df = DF_NORMAL;
    }
    return desc;
}
int gatherTest(int argc, char **argv, DataType dt)
{
    // input dim
    U32 iDim = atoi(argv[1]);
    U32 in = atoi(argv[2]);
    U32 ic = atoi(argv[3]);
    U32 ih = atoi(argv[4]);
    U32 iw = atoi(argv[5]);
    U32 indexDim = atoi(argv[6]);
    U32 index_n = atoi(argv[7]);
    U32 index_c = atoi(argv[8]);
    U32 index_h = atoi(argv[9]);
    U32 index_w = atoi(argv[10]);
    I32 axis = atoi(argv[11]);
    ArchInfo archInfo;
    archInfo.arch = MALI;

    GatherParamSpec p;
    p.axis = axis;
    p.element_level = false;

    TensorDesc inputDesc, indexDesc, outputDesc;
    inputDesc = createDesc(iDim, dt, in, ic, ih, iw);
    indexDesc = createDesc(indexDim, DT_I32, index_n, index_c, index_h, index_w);
    U32 inputLen = tensorNumElements(inputDesc);
    U8 *inputCpu = ut_input_v(inputLen, dt, UT_INIT_RANDOM);
    U32 indexLen = tensorNumElements(indexDesc);
    U8 *indexCpu = ut_input_v(inputLen, DT_I32, UT_INIT_RANDOM);
    I32 *val = (I32 *)indexCpu;
    axis = (axis + iDim) % iDim;
    U32 iDims[4] = {in, ic, ih, iw};
    for (U32 i = 0; i < indexLen; i++) {
        val[i] = rand() % iDims[axis];
    }

    Tensor inputTensorCpu, indexTensorCpu, outputTensorCpu;
    inputTensorCpu.resize(inputDesc);
    inputTensorCpu.alloc();
    UNI_MEMCPY(
        get_ptr_from_tensor(inputTensorCpu, CPU_GENERAL), inputCpu, tensorNumBytes(inputDesc));
    indexTensorCpu.resize(indexDesc);
    indexTensorCpu.alloc();
    UNI_MEMCPY(
        get_ptr_from_tensor(indexTensorCpu, CPU_GENERAL), indexCpu, tensorNumBytes(indexDesc));
    CHECK_STATUS(gather_infer_output_size(
        &inputTensorCpu, &indexTensorCpu, p, &outputTensorCpu, &UT_SERIAL_ARCHINFO));
    outputTensorCpu.alloc();
    U32 maxBytes = 0;
    CHECK_STATUS(gather_infer_forward_tmp_bytes(
        inputTensorCpu, indexTensorCpu, p, outputTensorCpu, &maxBytes, &UT_SERIAL_ARCHINFO));
    Tensor tmpTensorCpu;
    tmpTensorCpu.resize(tensor1d(DT_U8, maxBytes));
    tmpTensorCpu.alloc();
    CHECK_STATUS(gather(
        inputTensorCpu, indexTensorCpu, p, tmpTensorCpu, outputTensorCpu, &UT_SERIAL_ARCHINFO));

    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    MemoryType memType = OCLMem;
    if (archInfo.arch == QUALCOMM) {
        memType = OCLMemImg;
    }
    Tensor inputTensor = Tensor(memType);
    Tensor indexTensor = Tensor(OCLMem);
    Tensor outputTensor = Tensor(OCLMem);
    inputTensor.resize(inputDesc);
    indexTensor.resize(indexDesc);

    MaliPara maliPara;
    maliPara.handle = handle;
    archInfo.archPara = &maliPara;

    CHECK_STATUS(gather_infer_output_size(&inputTensor, &indexTensor, p, &outputTensor, &archInfo));
    outputDesc = outputTensor.get_desc();
    U8 *outputGpu = ut_input_v(tensorNumBytes(outputDesc), dt, UT_INIT_RANDOM);

    GCLMem_t output = alloc(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    GCLMem_t index = alloc(indexTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));

    Tensor tmpTensor = Tensor(OCLMem);
    U32 tmpBytes;
    CHECK_STATUS(gather_infer_forward_tmp_bytes(
        inputTensor, indexTensor, p, outputTensor, &maxBytes, &archInfo));
    tmpBytes = tensorNumBytes(inputDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(indexDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(outputDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(ocl_set_input(handle, input, inputDesc, inputCpu, tmpbuf, true));
    CHECK_STATUS(ocl_set_input(handle, index, indexDesc, indexCpu, tmpbuf, true));
    CHECK_STATUS(gather(inputTensor, indexTensor, p, tmpTensor, outputTensor, &archInfo));

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

    CHECK_STATUS(ocl_get_output(handle, output, outputDesc, outputGpu, tmpbuf, true));
    char buffer[150];
    char params[120];
    U32 on, oc, oh, ow;
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);
    sprintf(params, "(%u %u %u %u)->(%u %u %u %u)", in, ic, ih, iw, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "gather", params);
    double ops = ow * oh * oc * on;
    ut_log(dt, buffer, ops, time);

    ut_check_v(outputGpu, get_ptr_from_tensor(outputTensorCpu, CPU_GENERAL), on * oc * ow * oh, dt, 0.1);

    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(outputGpu);
    free(inputCpu);
    free(indexCpu);
    return 0;
}

int main(int argc, char **argv)
{
    gatherTest(argc, argv, DT_F16);
    return 0;
}
