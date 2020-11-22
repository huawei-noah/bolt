// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <string.h>
#include <vector>

#include "tensor_computing.h"
#include "ut_util.h"

int reshapeTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc > 4);
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 ih = atoi(argv[3]);
    U32 iw = atoi(argv[4]);
    ReshapeParamSpec p;
    p.shape_size = atoi(argv[5]);
    CHECK_REQUIREMENT(argc == 6 + p.shape_size);
    for (I32 i = 0; i < p.shape_size; i++) {
        p.shape_dims[i] = atoi(argv[6 + i]);
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

    Tensor outputTensor;
    CHECK_STATUS(reshape_infer_output_size(&inputTensor, p, &outputTensor, &archInfo));
    outputTensor.alloc();
    TensorDesc outDesc = outputTensor.get_desc();
    Tensor nullTensor;

    if (UT_CHECK) {
        CHECK_STATUS(reshape(inputTensor, nullTensor, outputTensor, &archInfo));

        CHECK_REQUIREMENT(tensorNumElements(outDesc) == len);
    }

    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(reshape(inputTensor, nullTensor, outputTensor, &archInfo));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    memset(params, 0, 120);
    sprintf(params, "(%u %u %u %u)=(", in, ic, ih, iw);
    for (I32 i = 0; i < p.shape_size; i++) {
        I32 index = 0;
        for (; index < 120; index++) {
            if (params[index] == '\0') {
                break;
            }
        }
        if (i != p.shape_size - 1) {
            sprintf(params + index, "%d ", outDesc.dims[outDesc.nDims - 1 - i]);
        } else {
            sprintf(params + index, "%d)", outDesc.dims[outDesc.nDims - 1 - i]);
        }
    }
    sprintf(buffer, "%20s, %80s", "Reshape", params);
    double ops = len;
    ut_log(dt, buffer, ops, time);

    free(input);

    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    reshapeTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    reshapeTest(argc, argv, DT_F32);
#endif
    return 0;
}
