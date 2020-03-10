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

int transposeTest(int argc, char** argv, DataType dt) {
    CHECK_REQUIREMENT(argc == 9);
    U32 in  = atoi(argv[1]);
    U32 ic  = atoi(argv[2]);
    U32 ih  = atoi(argv[3]);
    U32 iw  = atoi(argv[4]);
    std::vector<U32> dim(4, 0);
    std::vector<U32> inv_dim(4, 0);
    for (int i = 0; i < 4; i++) {
        I32 value = atoi(argv[5+i]);
        dim[i] = value;
        inv_dim[value] = i;
    }

    DataFormat df = DF_NCHW;
    TensorDesc in_desc = tensor4df(dt, df, in, ic, ih, iw);
    TensorDesc out_1_desc;
    TensorDesc out_2_desc;

    CHECK_STATUS(transpose_infer_output_size(in_desc, &out_1_desc, dim.data()));
    CHECK_STATUS(transpose_infer_output_size(out_1_desc, &out_2_desc, inv_dim.data()));

    U32 len = tensorNumElements(in_desc);
    U8* input = ut_input_v(len, dt, UT_INIT_RANDOM);
    U8* out_1 = ut_input_v(len, dt, UT_INIT_RANDOM);
    U8* out_2 = ut_input_v(len, dt, UT_INIT_RANDOM);

    if (UT_CHECK) {
        CHECK_STATUS(transpose(in_desc, input, out_1_desc, out_1, dim.data(), UT_ARCH));

        CHECK_STATUS(transpose(out_1_desc, out_1, out_2_desc, out_2, inv_dim.data(), UT_ARCH));

        // check
        ut_check_v(input, out_2, len, dt, 0.0001, __FILE__, __LINE__);
    }

    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(transpose(in_desc, input, out_1_desc, out_1, dim.data(), UT_ARCH));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    U32 on = 0;
    U32 oc = 0;
    U32 oh = 0;
    U32 ow = 0;
    CHECK_STATUS(tensor4dGet(out_1_desc, &dt, &df, &on, &oc, &oh, &ow));
    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u)=(%u %u %u %u)",
                    in, ic, ih, iw,
                    on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "Transpose", params);
    double ops = len;
    ut_log(dt, buffer, ops, time);

    free(input);
    free(out_1);
    free(out_2);

    return 0;
}


int main(int argc, char** argv) {
#ifdef _USE_FP16
    transposeTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    transposeTest(argc, argv, DT_F32);
#endif
    return 0;
}
