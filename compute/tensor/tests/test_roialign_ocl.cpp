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

int roialignTest(int argc, char *argv[], DataType dt)
{
    U32 in, ic, ih, iw;
    U32 on, oc, oh, ow;
    U32 roiNum, roiSize;
    U32 batchIndex;
    U32 sample_ratio;
    F32 spatial_scale;

    ArchInfo archInfo;
    archInfo.arch = MALI;
    if (gcl_check_device_qualcomm(OCLContext::getInstance().handle.get())) {
        archInfo.arch = QUALCOMM;
    }

    in = 1;
    ic = 4;
    ih = 8;
    iw = 8;
    oh = 2;
    ow = 2;
    roiNum = 4;
    roiSize = 4;
    batchIndex = 1;
    sample_ratio = 0;
    spatial_scale = 1.0;
    U32 useInputC4 = 0;

    if (argc >= 10) {
        in = atoi(argv[1]);
        ic = atoi(argv[2]);
        ih = atoi(argv[3]);
        iw = atoi(argv[4]);
        oh = atoi(argv[5]);
        ow = atoi(argv[6]);
        roiNum = atoi(argv[7]);
        sample_ratio = atoi(argv[8]);
        spatial_scale = atof(argv[9]);
        if (argc >= 11) {
            useInputC4 = atoi(argv[10]);
        }
    }

    RoIAlignParamSpec p;
    p.trans_mode = COORDINATE_TRANS_HALF_PIXEL;
    p.mode = POOLING_MEAN;
    p.output_w = ow;
    p.output_h = oh;
    p.sampling_ratio = sample_ratio;
    p.spatial_scale = spatial_scale;

    U32 num = 2;
    std::vector<void *> inputCpu(num);
    std::vector<Tensor> inTensorsCpu(num);
    std::vector<Tensor *> inTensorPtrCpu(num);
    TensorDesc inputDesc = tensor4df(dt, DF_NCHW, in, ic, ih, iw);
    TensorDesc roiDesc = tensor2df(dt, DF_NORMAL, roiNum, roiSize);
    Tensor outTensorCpu;
    for (U32 i = 0; i < num; i++) {
        if (i == 0) {
            inputCpu[i] = (void *)ut_input_v(tensorNumElements(inputDesc), dt, UT_INIT_RANDOM);
            inTensorsCpu[i].resize(inputDesc);
        } else if (i == 1) {
            inputCpu[i] = (void *)ut_input_v(tensorNumElements(roiDesc), dt, UT_INIT_ZERO);
            F16 *val = (F16 *)inputCpu[i];
            for (U32 i = 0; i < roiNum; i++) {
                val[i * 4] = ((rand() % 1024) / 1024.0) * iw / spatial_scale;
                val[i * 4 + 1] = ((rand() % 1024) / 1024.0) * ih / spatial_scale;
                val[i * 4 + 2] = val[i * 4] + ((rand() % 1024) / 1024.0) * iw / spatial_scale;
                val[i * 4 + 3] = val[i * 4 + 1] + ((rand() % 1024) / 1024.0) * ih / spatial_scale;
            }
            inTensorsCpu[i].resize(roiDesc);
        }
        inTensorsCpu[i].alloc();
        UNI_MEMCPY(get_ptr_from_tensor(inTensorsCpu[i], CPU_GENERAL), inputCpu[i],
            tensorNumBytes(inTensorsCpu[i].get_desc()));
        inTensorPtrCpu[i] = &inTensorsCpu[i];
    }
    CHECK_STATUS(roialign_infer_output_size(inTensorPtrCpu, p, &outTensorCpu, &UT_SERIAL_ARCHINFO));
    outTensorCpu.alloc();

    Tensor tmpTensorCpu;
    CHECK_STATUS(roialign(inTensorsCpu, p, tmpTensorCpu, outTensorCpu, &UT_SERIAL_ARCHINFO));

    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    if (useInputC4) {
        inputDesc.df = DF_NCHWC4;
    }
    std::vector<Tensor> inTensors(num);
    std::vector<Tensor *> inTensorPtr;
    Tensor outTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    MemoryType memType = OCLMem;
    for (U32 i = 0; i < num; i++) {
        memType = (archInfo.arch == QUALCOMM && i == 0) ? OCLMemImg : OCLMem;
        Tensor tensor = Tensor(memType);
        if (i == 0) {
            tensor.resize(inputDesc);
        } else if (i == 1) {
            tensor.resize(roiDesc);
        }
        inTensors[i] = tensor;
        inTensorPtr.push_back(&inTensors[i]);
    }

    MaliPara maliPara;
    maliPara.handle = handle;
    archInfo.archPara = &maliPara;
    CHECK_STATUS(roialign_infer_output_size(inTensorPtr, p, &outTensor, &archInfo));
    TensorDesc outputDesc = outTensor.get_desc();
    U8 *output_gpu = ut_input_v(tensorNumBytes(outputDesc), dt, UT_INIT_RANDOM);

    U32 maxBytes = 0;
    U32 tmpBytes;
    CHECK_STATUS(roialign_infer_forward_tmp_bytes(inTensors[0], outTensor, &tmpBytes, &archInfo));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(inputDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(roiDesc);
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
        TensorDesc desc = (i == 0) ? inputDesc : roiDesc;
        CHECK_STATUS(ocl_set_input(handle, inputVec[i], desc, (U8 *)inputCpu[i], tmpbuf, true));
    }
    CHECK_STATUS(roialign(inTensors, p, tmpTensor, outTensor, &archInfo));

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

    char buffer[150];
    char params[120];
    on = outputDesc.dims[outputDesc.nDims - 1];
    oc = outputDesc.dims[outputDesc.nDims - 2];
    sprintf(params, "(%u %u %u %u) * (%u %u) * (%u) * (%u %u) = (%u %u %u %u)", in, ic, ih, iw,
        roiNum, roiSize, batchIndex, oh, ow, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "Roialign", params);
    double ops = 1.0 * tensorNumElements(outputDesc);
    ut_log(dt, buffer, ops, time);

    CHECK_STATUS(ocl_get_output(handle, output, outputDesc, output_gpu, tmpbuf, true));
    ut_check_v(output_gpu, get_ptr_from_tensor(outTensorCpu, CPU_GENERAL),
        tensorNumElements(outputDesc), dt, 0.3);
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
    roialignTest(argc, argv, DT_F16);
    return 0;
}
