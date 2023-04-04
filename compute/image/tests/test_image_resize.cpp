// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "image.h"
#include "ut_util.h"

int resizeTest(int argc, char *argv[], DataType dt)
{
    CHECK_REQUIREMENT(argc == 9);
    // in data
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 ih = atoi(argv[3]);
    U32 iw = atoi(argv[4]);
    // output
    U32 on = atoi(argv[5]);
    U32 oc = atoi(argv[6]);
    U32 oh = atoi(argv[7]);
    U32 ow = atoi(argv[8]);
    CHECK_REQUIREMENT(in == 1 && on == 1);

    ArchInfo archInfo;
    archInfo.arch = UT_ARCH;

    ResizeParamSpec p;
    p.mode = RESIZE_LINEAR;
    p.trans_mode = COORDINATE_TRANS_ASYMMETRIC;
    p.num_sizes = 0;
    p.num_scales = 4;
    p.scales[0] = oh;
    p.scales[1] = ow;
    p.scales[2] = (F32)oh / (F32)ih;
    p.scales[3] = (F32)ow / (F32)iw;

    // setup input
    TensorDesc inputDesc = tensor4df(dt, DF_NCHW, in, ic, ih, iw);
    U8 *input = ut_input_v(in * ic * ih * iw, dt, UT_INIT_RANDOM);
    Tensor inputTensor = Tensor::alloc_sized<CPUMem>(inputDesc);
    UNI_MEMCPY(get_ptr_from_tensor(inputTensor, UT_ARCH), input, tensorNumBytes(inputDesc));

    // setup output
    Tensor outputTensor;
    CHECK_STATUS(resize_infer_output_size(&inputTensor, p, &outputTensor, &archInfo));
    TensorDesc outputDesc = outputTensor.get_desc();
    CHECK_REQUIREMENT(tensorNumElements(outputDesc) == on * oc * oh * ow);
    outputTensor.alloc();
    Tensor outputTensorRef = Tensor::alloc_sized<CPUMem>(outputDesc);
    U32 cpuTmpBytes = 0, cpuTmpBytesSerial = 0;
    CHECK_STATUS(
        resize_infer_forward_tmp_bytes(inputTensor, p, outputTensor, &cpuTmpBytes, &archInfo));
    CHECK_STATUS(resize_infer_forward_tmp_bytes(
        inputTensor, p, outputTensorRef, &cpuTmpBytesSerial, &UT_SERIAL_ARCHINFO));
    Tensor tmpTensor = Tensor::alloc_sized<CPUMem>(tensor1d(DT_I8, cpuTmpBytes));
    Tensor tmpTensorSerial = Tensor::alloc_sized<CPUMem>(tensor1d(DT_I8, cpuTmpBytesSerial));
    if (UT_CHECK) {
        CHECK_STATUS(resize(inputTensor, p, tmpTensor, outputTensor, &archInfo));

        // naive implement
        CHECK_STATUS(resize(inputTensor, p, tmpTensorSerial, outputTensorRef, &UT_SERIAL_ARCHINFO));

        // check
        ut_check_v(get_ptr_from_tensor(outputTensor, UT_ARCH),
            get_ptr_from_tensor(outputTensorRef, UT_ARCH), outputTensor.length(), dt, 0.05);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(resize(inputTensor, p, tmpTensor, outputTensor, &archInfo));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u)=>(%u %u %u %u)", in, ic, ih, iw, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "Resize", params);
    double ops = 15.0 * on * oc * oh * ow;
    ut_log(dt, buffer, ops, time);

    free(input);
    return 0;
}

int main(int argc, char *argv[])
{
#ifdef _USE_FP16
    resizeTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    resizeTest(argc, argv, DT_F32);
#endif
    return 0;
}
