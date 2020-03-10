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

#ifdef _USE_INT8
int int8PoolingTest(int argc, char **argv, DataType dt) {
    CHECK_REQUIREMENT(argc == 15);
    // in data
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 ih = atoi(argv[3]);
    U32 iw = atoi(argv[4]);
    // weight
    U32 fn = atoi(argv[5]);
    U32 fc = atoi(argv[6]);
    U32 fh = atoi(argv[7]);
    U32 fw = atoi(argv[8]);
    // stride & padding
    U32 stride  = atoi(argv[9]);
    U32 padding = atoi(argv[10]);
    // output
    U32 on = atoi(argv[11]);
    U32 oc = atoi(argv[12]);
    U32 oh = atoi(argv[13]);
    U32 ow = atoi(argv[14]);
    CHECK_REQUIREMENT(in == 1 && fn == 1 && fc == 1);
    CHECK_REQUIREMENT(ic == oc && ic % 8 == 0);

    PoolingDesc pooling_desc;
    pooling_desc.pm = POOLING_MEAN;
    
    pooling_desc.stride_h = stride;
    pooling_desc.stride_w = stride;
    pooling_desc.padding_top = padding;
    pooling_desc.padding_bottom = padding;
    pooling_desc.padding_left = padding;
    pooling_desc.padding_right = padding;
    pooling_desc.kernelSize_h = fh;
    pooling_desc.kernelSize_w = fw;
    pooling_desc.rm = CEIL;

    TensorDesc input_desc= tensor4df(DT_I8, DF_NCHWC8, in, ic, ih, iw);
    TensorDesc in_desc_ref = input_desc;
    in_desc_ref.dt = dt;

    TensorDesc output_desc;
    CHECK_STATUS(pooling_infer_output_size(input_desc, pooling_desc, &output_desc, UT_ARCH));
    U32 input_len = tensorNumElements(input_desc);
    U32 output_len = tensorNumElements(output_desc);
    CHECK_REQUIREMENT(input_len == in*ic*ih*iw && output_len == on*oc*oh*ow);

    U8* input_ref  = ut_input_v(input_len, dt, UT_INIT_RANDOM);
    INT8* input = (INT8*)ut_input_v(input_len, DT_I8, UT_INIT_ZERO);
    F16 scales[2];
    quantize_tensor(in_desc_ref, input_ref, &input_desc, input, scales);

    INT8* output = (INT8*)ut_input_v(output_len, DT_I8, UT_INIT_ZERO);
    U8* out_d = ut_input_v(output_len, dt, UT_INIT_ZERO);
    U8* output_ref = ut_input_v(output_len, dt, UT_INIT_ZERO);

    if(UT_CHECK){
        CHECK_STATUS(pooling(input_desc, input,
                             pooling_desc, scales,
                             output_desc, output,
                             UT_ARCH));

        for (U32 i=0; i<output_len; i++) {
            switch (dt) {
#ifdef _USE_FP32
                case DT_F32:
                    ((F32*)out_d)[i] = output[i] / scales[1];
                    break;
#endif
#ifdef _USE_FP16
                case DT_F16:
                    ((F16*)out_d)[i] = output[i] / scales[1];
                    break;
#endif
                default:
                    break;
            }
        }

        CHECK_STATUS(pooling(in_desc_ref, input_ref,
                             pooling_desc, nullptr,
                             output_desc, output_ref,
                             CPU_GENERAL));

        // check
        ut_check_v(out_d, output_ref, output_len, dt, 0.05, __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for(int iter=0; iter<UT_LOOPS; iter++){
        CHECK_STATUS(pooling(input_desc, input,
                             pooling_desc, scales,
                             output_desc, output,
                             UT_ARCH));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u)+(%u %u %u %u)/(%u %u)=(%u %u %u %u)",
                    in, ic, ih, iw,
                    fn, fc, fh, fw,
                    stride, padding,
                    on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "Pooling", params);
    double ops = 1.0 * on * oc * oh * ow * fh * fw;
    ut_log(DT_I8, buffer, ops, time);

    free(input);
    free(input_ref);
    free(output);
    free(out_d);
    free(output_ref);

    return 0;
}
#endif

int main(int argc, char** argv) {
#ifdef _USE_INT8
    int8PoolingTest(argc, argv, DT_F16);
#endif
    return 0;
}
