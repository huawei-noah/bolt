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

int sliceTest(int argc, char** argv, DataType dt) {
    CHECK_REQUIREMENT(argc > 2);
    I32 num = atoi(argv[1]);
    CHECK_REQUIREMENT(argc == 2+4+1+num-1);
    U32 in  = atoi(argv[2]);
    U32 ic  = atoi(argv[3]);
    U32 ih  = atoi(argv[4]);
    U32 iw  = atoi(argv[5]);
    U32 axis= atoi(argv[6]);
    std::vector<U32> slice_point(num);
    for (I32 i = 0; i < num-1; i++) {
        slice_point[i] = atoi(argv[7+i]);
    }

    DataFormat df = DF_NCHW;
    TensorDesc in_desc = tensor4df(dt, df, in, ic, ih, iw);
    std::vector<TensorDesc> out_desc(num);

    CHECK_STATUS(slice_infer_output_size(in_desc, &out_desc, axis, slice_point.data()));
    std::vector<void*> output(num);
    for (I32 i = 0; i < num; i++) {
        output[i] = (void*)ut_input_v(tensorNumElements(out_desc[i]), dt, UT_INIT_ZERO);
    }

    U32 len = tensorNumElements(in_desc);
    U8* input = ut_input_v(len, dt, UT_INIT_RANDOM);

    if (UT_CHECK) {
        CHECK_STATUS(slice(in_desc, input, out_desc, &output, UT_ARCH));

        U32 tmp = 0;
        for (I32 i = 0; i < num; i++) {
            tmp += tensorNumElements(out_desc[i]);
        }
        CHECK_REQUIREMENT(tmp == len);
    }

    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(slice(in_desc, input, out_desc, &output, UT_ARCH));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u)=(%u %u %u %u)/%u",
                    in, ic, ih, iw,
                    in, ic, ih, iw, num);
    sprintf(buffer, "%20s, %80s", "Slice", params);
    double ops = num * len;
    ut_log(dt, buffer, ops, time);

    free(input);

    return 0;
}


int main(int argc, char** argv) {
#ifdef _USE_FP16
    sliceTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    sliceTest(argc, argv, DT_F32);
#endif
    return 0;
}
