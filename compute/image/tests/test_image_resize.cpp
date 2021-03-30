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
    ArchInfo archInfo;
    archInfo.arch = UT_ARCH;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;

    CHECK_REQUIREMENT(in == 1 && on == 1);

    TensorDesc inputDesc, outputDesc;
    inputDesc = tensor4df(dt, DF_NCHW, in, ic, ih, iw);

    DataType paramDT = DT_F32;
    F32 scales[2];
    scales[0] = (F32)oh / (F32)ih;
    scales[1] = (F32)ow / (F32)iw;

    // setup input
    U8 *input = ut_input_v(in * ic * ih * iw, dt, UT_INIT_RANDOM);
    Tensor inputTensor;
    inputTensor.resize(inputDesc);
    inputTensor.alloc();
    memcpy(get_ptr_from_tensor(inputTensor, UT_ARCH), input, tensorNumBytes(inputDesc));

    // setup output
    U32 outputBytes;
    Tensor outputTensor;
    CHECK_STATUS(resize_infer_output_size(
        &inputTensor, paramDT, scales, &outputTensor, &outputBytes, &archInfo));
    outputDesc = outputTensor.get_desc();
    CHECK_REQUIREMENT(tensorNumElements(outputDesc) == on * oc * oh * ow);
    outputTensor.alloc();
    Tensor outputTensorRef = Tensor::alloc_sized<CPUMem>(outputDesc);
    Tensor tmpTensor = Tensor::alloc_sized<CPUMem>(tensor1d(DT_U8, 8 * tensorNumBytes(inputDesc)));

    ResizeParamSpec p;
    p.mode = LINEAR;
    if (UT_CHECK) {
        CHECK_STATUS(resize(inputTensor, tmpTensor, outputTensor, p, &archInfo));

        // naive implement
        CHECK_STATUS(resize(inputTensor, tmpTensor, outputTensorRef, p, &archInfo_org));

        // check
        ut_check_v(get_ptr_from_tensor(outputTensor, UT_ARCH),
            get_ptr_from_tensor(outputTensorRef, UT_ARCH), outputTensor.length(), dt, 0.05,
            __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(resize(inputTensor, tmpTensor, outputTensor, p, &archInfo));
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
