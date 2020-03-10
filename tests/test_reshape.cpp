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

int reshapeTest(int argc, char** argv, DataType dt) {
    CHECK_REQUIREMENT(argc > 4);
    U32 in  = atoi(argv[1]);
    U32 ic  = atoi(argv[2]);
    U32 ih  = atoi(argv[3]);
    U32 iw  = atoi(argv[4]);
    I32 shape_size = atoi(argv[5]);
    CHECK_REQUIREMENT(argc == 6+shape_size);
    std::vector<I32> shape(shape_size);
    for (I32 i = 0; i < shape_size; i++) {
        shape[i] = atoi(argv[6+i]);
    }

    DataFormat df = DF_NCHW;
    TensorDesc in_desc = tensor4df(dt, df, in, ic, ih, iw);
    TensorDesc out_desc;

    CHECK_STATUS(reshape_infer_output_size(in_desc, &out_desc, shape.data(), shape_size));

    U32 len = tensorNumElements(in_desc);
    U8* input  = ut_input_v(len, dt, UT_INIT_RANDOM);
    U8* output = ut_input_v(len, dt, UT_INIT_RANDOM);

    if (UT_CHECK) {
        CHECK_STATUS(reshape(in_desc, input, out_desc, output, UT_ARCH));

        CHECK_REQUIREMENT(tensorNumElements(out_desc) == len);
    }

    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(reshape(in_desc, input, out_desc, output, UT_ARCH));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    memset(params, 0, 120);
    sprintf(params, "(%u %u %u %u)=(",
                    in, ic, ih, iw);
    for(I32 i = 0; i < shape_size; i++) {
        I32 index = 0;
        for (; index < 120; index++) {
            if (params[index] == '\0') {
                break;
            }
        }
        if (i != shape_size-1) {
            sprintf(params+index, "%d ", out_desc.dims[out_desc.nDims-1-i]);
        }
        else {
            sprintf(params+index, "%d)", out_desc.dims[out_desc.nDims-1-i]);
        }
    }
    sprintf(buffer, "%20s, %80s", "Reshape", params);
    double ops = len;
    ut_log(dt, buffer, ops, time);

    free(input);
    free(output);

    return 0;
}


int main(int argc, char** argv) {
#ifdef _USE_FP16
    reshapeTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    reshapeTest(argc, argv, DT_F32);
#endif
    return 0;
}
