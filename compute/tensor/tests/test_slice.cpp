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

int sliceTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc > 2);
    I32 num = atoi(argv[1]);
    CHECK_REQUIREMENT(argc == 2 + 4 + 1 + num - 1);
    U32 in = atoi(argv[2]);
    U32 ic = atoi(argv[3]);
    U32 ih = atoi(argv[4]);
    U32 iw = atoi(argv[5]);
    SliceParamSpec p;
    p.axis = atoi(argv[6]);
    p.slice_size = num - 1;
    for (U32 i = 0; i < p.slice_size; i++) {
        p.slice_points[i] = atoi(argv[7 + i]);
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

    std::vector<Tensor> outputTensors(num);
    std::vector<Tensor *> outputTensorsPtr(num);
    for (I32 i = 0; i < num; i++) {
        outputTensorsPtr[i] = &outputTensors[i];
    }
    CHECK_STATUS(slice_infer_output_size(&inputTensor, p, outputTensorsPtr, &archInfo));
    for (I32 i = 0; i < num; i++) {
        outputTensors[i].alloc();
    }

    Tensor tmpTensor;
    if (UT_CHECK) {
        CHECK_STATUS(slice(inputTensor, p, tmpTensor, outputTensors, &archInfo));

        U32 tmp = 0;
        for (I32 i = 0; i < num; i++) {
            tmp += outputTensors[i].length();
        }
        CHECK_REQUIREMENT(tmp == len);
    }

    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(slice(inputTensor, p, tmpTensor, outputTensors, &archInfo));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u)=(%u %u %u %u)/%u", in, ic, ih, iw, in, ic, ih, iw, num);
    sprintf(buffer, "%20s, %80s", "Slice", params);
    double ops = num * len;
    ut_log(dt, buffer, ops, time);

    free(input);

    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    sliceTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    sliceTest(argc, argv, DT_F32);
#endif
    return 0;
}
