// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <vector>

#include "tensor_computing.h"
#include "ut_util.h"

int transposeTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc == 9);
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 ih = atoi(argv[3]);
    U32 iw = atoi(argv[4]);
    TransposeParamSpec p, p_inv;
    p.trans_size = 4;
    p_inv.trans_size = 4;
    for (int i = 0; i < 4; i++) {
        I32 value = atoi(argv[5 + i]);
        p.trans_dims[i] = value;
        p_inv.trans_dims[value] = i;
    }
    ArchInfo archInfo;
    archInfo.arch = UT_ARCH;

    DataFormat df = DF_NCHW;
    TensorDesc inDesc = tensor4df(dt, df, in, ic, ih, iw);
    U32 len = tensorNumElements(inDesc);
    U8 *input = ut_input_v(len, dt, UT_INIT_RANDOM);
    Tensor inputTensor;
    inputTensor.resize(inDesc);
    inputTensor.alloc();
    memcpy(get_ptr_from_tensor(inputTensor, UT_ARCH), input, tensorNumBytes(inDesc));

    Tensor outputTensor1;
    Tensor outputTensor2;
    CHECK_STATUS(transpose_infer_output_size(&inputTensor, p, &outputTensor1, &archInfo));
    CHECK_STATUS(transpose_infer_output_size(&outputTensor1, p_inv, &outputTensor2, &archInfo));
    outputTensor1.alloc();
    outputTensor2.alloc();
    Tensor blankTensor;

    if (UT_CHECK) {
        CHECK_STATUS(transpose(inputTensor, p, blankTensor, outputTensor1, &archInfo));

        CHECK_STATUS(transpose(outputTensor1, p_inv, blankTensor, outputTensor2, &archInfo));

        // check
        ut_check_v(input, get_ptr_from_tensor(outputTensor2, UT_ARCH), len, dt, 0.0001, __FILE__,
            __LINE__);
    }

    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(transpose(inputTensor, p, blankTensor, outputTensor1, &archInfo));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    U32 on = 0;
    U32 oc = 0;
    U32 oh = 0;
    U32 ow = 0;
    CHECK_STATUS(tensor4dGet(outputTensor1.get_desc(), &dt, &df, &on, &oc, &oh, &ow));
    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u)=(%u %u %u %u)", in, ic, ih, iw, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "Transpose", params);
    double ops = len;
    ut_log(dt, buffer, ops, time);

    free(input);

    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    transposeTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    transposeTest(argc, argv, DT_F32);
#endif
    return 0;
}
