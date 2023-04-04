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

int reductionTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc >= 6);
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 ih = atoi(argv[3]);
    U32 iw = atoi(argv[4]);
    ReductionParamSpec p;
    p.num_axes = atoi(argv[5]);
    for (int i = 0; i < p.num_axes; i++) {
        p.axes[i] = atoi(argv[6 + i]);
    }
    p.mode = REDUCTION_MEAN;
    p.coeff = 1.0;
    p.keep_dim = true;
    DataFormat df = DF_NCHW;
    TensorDesc maskDesc;
    maskDesc.nDims = 0;

    TensorDesc inDesc = tensor4df(dt, df, in, ic, ih, iw);
    U8 *input = ut_input_v(tensorNumElements(inDesc), dt, UT_INIT_RANDOM);
    Tensor inputTensor;
    inputTensor.resize(inDesc);
    inputTensor.alloc();
    UNI_MEMCPY(get_ptr_from_tensor(inputTensor, CPU_GENERAL), input, tensorNumBytes(inDesc));

    Tensor maskTensor;
    maskTensor.resize(maskDesc);
    Tensor outputTensor;
    Tensor outputTensorRef;
    CHECK_STATUS(
        reduction_infer_output_size(&inputTensor, maskTensor, p, &outputTensor, &UT_CPU_ARCHINFO));
    outputTensor.alloc();
    outputTensorRef.resize(outputTensor.get_desc());
    outputTensorRef.alloc();

    U32 tmpBytes;
    CHECK_STATUS(reduction_infer_forward_tmp_bytes(
        inputTensor, p, outputTensor, &tmpBytes, &UT_CPU_ARCHINFO));
    Tensor tmpTensor;
    tmpTensor.resize(tensor1d(dt, tmpBytes));
    tmpTensor.alloc();

    if (UT_CHECK) {
        CHECK_STATUS(
            reduction(inputTensor, maskTensor, p, tmpTensor, outputTensor, &UT_CPU_ARCHINFO));

        // naive implement
        CHECK_STATUS(
            reduction(inputTensor, maskTensor, p, tmpTensor, outputTensorRef, &UT_SERIAL_ARCHINFO));

        // check
        ut_check_v(get_ptr_from_tensor(outputTensor, CPU_GENERAL),
            get_ptr_from_tensor(outputTensorRef, CPU_GENERAL), outputTensor.length(), dt, 1);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(
            reduction(inputTensor, maskTensor, p, tmpTensor, outputTensor, &UT_CPU_ARCHINFO));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    U32 on = 1, oc = 1, oh = 1, ow = 1;
    CHECK_STATUS(tensor4dGet(outputTensor.get_desc(), &dt, &df, &on, &oc, &oh, &ow));
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u) %d =(%u %u %u %u)", in, ic, ih, iw, p.num_axes, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "Reduction", params);
    double ops = 1.0 * in * ic * ih * iw;
    ut_log(dt, buffer, ops, time / UT_LOOPS);

    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    reductionTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    reductionTest(argc, argv, DT_F32);
#endif
    return 0;
}
