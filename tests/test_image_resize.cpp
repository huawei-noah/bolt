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
#include "image.h"
#include "ut_util.h"


int resizeTest(int argc, char* argv[], DataType dt)
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
    CHECK_REQUIREMENT(ic % 8 == 0 && oc % 8 == 0);

    TensorDesc inputDesc, outputDesc;
    ResizeDesc resizeDesc;
    inputDesc = tensor4df(dt, DF_NCHWC8, in, ic, ih, iw);

    resizeDesc.paramDT = DT_F32;
    F32 scales[2];
    scales[0] = (F32)oh / (F32)ih;
    scales[1] = (F32)ow / (F32)iw;

    // setup input, filter
    U8 *input  = ut_input_v(in*ic*ih*iw, dt, UT_INIT_RANDOM);
    U8 *input_ref  = ut_input_v(in*ic*ih*iw, dt, UT_INIT_ZERO);
    memcpy(input_ref, input, bytesOf(dt)*in*ic*ih*iw);

    // setup output
    U32 outputBytes;
    CHECK_STATUS(resize_infer_output_size(inputDesc, resizeDesc, scales, &outputDesc, &outputBytes));
    CHECK_REQUIREMENT(tensorNumElements(outputDesc) == on*oc*oh*ow);
    U32 output_size = outputBytes / bytesOf(dt);
    U8 *output     = ut_input_v(output_size, dt, UT_INIT_ZERO);
    U8 *output_ref = ut_input_v(output_size, dt, UT_INIT_ZERO);

    if (UT_CHECK) {
        CHECK_STATUS(resize(inputDesc, input,
                            outputDesc, output,
                            UT_ARCH));

        // naive implement
        CHECK_STATUS(resize(inputDesc, input_ref,
                            outputDesc, output_ref,
                            CPU_GENERAL));

        // check
        ut_check_v(output, output_ref, output_size, dt, 0.05, __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(resize(inputDesc, input_ref,
                            outputDesc, output_ref,
                            CPU_GENERAL));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u)=>(%u %u %u %u)",
                    in, ic, ih, iw,
                    on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "Resize", params);
    double ops = 15.0 * on * oc * oh * ow;
    ut_log(dt, buffer, ops, time);

    free(input);
    free(output);
    free(input_ref);
    free(output_ref);
    return 0;
}

int main(int argc, char* argv[])
{
#ifdef _USE_FP16
    resizeTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    resizeTest(argc, argv, DT_F32);
#endif
    return 0;
}
