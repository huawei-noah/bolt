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
#include "tensor_computing.h"
#include "ut_util.h"

int bnnConvolutionTest(int argc, char* argv[], DataType dt) {
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

    CHECK_REQUIREMENT(in == 1 && on == 1);

    DataType fdt = DT_BIN11;  // Use dt to distinguish DoReFa and XNOR
    ActivationMode am = ACTIVATION_NULL;

    TensorDesc inputDesc = tensor4df(dt, DF_NCHWC8, in, ic, ih, iw);
    TensorDesc filterDesc = tensor4df(fdt, DF_NCHW, oc, ic, fh, fw);
    TensorDesc scaleDesc = tensor1d(dt, oc);
    TensorDesc biasDesc = tensor1d(dt, oc);
    ConvolutionDesc convDesc;
    convDesc.stride_h = stride;
    convDesc.stride_w = stride;
    convDesc.padding_top = padding;
    convDesc.padding_bottom = padding;
    convDesc.padding_left = padding;
    convDesc.padding_right = padding;
    convDesc.dilatedRate_h = 1;
    convDesc.dilatedRate_w = 1;

    // setup input, filter, bias
    U8 *input  = ut_input_v(in*ic*ih*iw, dt, UT_INIT_RANDOM);
    if (fdt == DT_BIN01) {
        for (U32 i = 0; i < in*ic*ih*iw; i++) {
            switch (dt) {
#ifdef _USE_FP16
                case DT_F16:
                    ((F16*)input)[i] += 0.5;
                    break;
#endif
#ifdef _USE_FP32
                case DT_F32:
                    ((F32*)input)[i] += 0.5;
                    break;
#endif
                default:
                    break;
            }
        }
    }
   
    BIN8 *filter     = (BIN8*)ut_input_v(fn*fc*fh*fw/8, fdt, UT_INIT_POS);
    U8 *scale        = ut_input_v(oc, dt, UT_INIT_RANDOM);
    U8 *bias         = ut_input_v(oc, dt, UT_INIT_RANDOM);
    U8 *input_ref    = ut_input_v(in*ic*ih*iw, dt, UT_INIT_ZERO);
    BIN8 *filter_ref = (BIN8*)ut_input_v(fn*fc*fh*fw/8, fdt, UT_INIT_ZERO);
    memcpy(input_ref,  input,  bytesOf(dt)*in*ic*ih*iw);
    memcpy(filter_ref, filter, sizeof(BIN8)*fn*fc*fh*fw/8);

    // setup output, bias
    U32 outputBytes;
    TensorDesc outputDesc;
    CHECK_STATUS(convolution_infer_output_size(inputDesc, filterDesc, convDesc, &outputDesc, dt, &outputBytes, UT_ARCH));
    U32 output_size = outputBytes / bytesOf(dt);

    U8 *output     = ut_input_v(output_size, dt, UT_INIT_ZERO);
    U8 *output_ref = ut_input_v(output_size, dt, UT_INIT_ZERO);

    // setup alg
    ConvolutionPolicy policy = CONVOLUTION_FASTEST;
    ConvolutionForwardAlgorithm alg = CONVOLUTION_ALGORITHM_NULL;
    CHECK_STATUS(convolution_infer_forward_algorithm(inputDesc, filterDesc, outputDesc, convDesc, policy, &alg, fdt, am, UT_ARCH));

    // setup tmp
    U32 tmpBytes;
    CHECK_STATUS(convolution_infer_forward_tmp_bytes(inputDesc, filterDesc, outputDesc, convDesc, alg, &tmpBytes, UT_ARCH));

    BIN8 *tmp     = (BIN8*)ut_input_v(tmpBytes/sizeof(BIN8), fdt, UT_INIT_ZERO);

    // setup filter trans
    U32 ftmBytes;
    CHECK_STATUS(convolution_transform_filter_bytes(filterDesc, alg, &ftmBytes, UT_ARCH));
    BIN8 *ftm     = (BIN8*)ut_input_v(ftmBytes/sizeof(BIN8), fdt, UT_INIT_ZERO);
    // trans filter
    TensorDesc ftmDesc;
    CHECK_STATUS(convolution_transform_filter(filterDesc, filter, alg, &ftmDesc, ftm, UT_ARCH));

    if (UT_CHECK) {
        CHECK_STATUS(convolution(inputDesc, input,
                                 ftmDesc, ftm,
                                 convDesc, alg,
                                 scaleDesc, scale,
                                 biasDesc, bias,
                                 tmpBytes, tmp,
                                 outputDesc, output,
                                 am, UT_ARCH));

        // naive implement
        CHECK_STATUS(convolution(inputDesc, input_ref,
                                 filterDesc, filter,
                                 convDesc, alg,
                                 scaleDesc, scale,
                                 biasDesc, bias,
                                 tmpBytes, tmp,
                                 outputDesc, output_ref,
                                 am, CPU_GENERAL));

        // check
        ut_check_v(output, output_ref, output_size, dt, 1, __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(convolution(inputDesc, input,
                                 ftmDesc, ftm,
                                 convDesc, alg,
                                 scaleDesc, scale,
                                 biasDesc, bias,
                                 tmpBytes, tmp,
                                 outputDesc, output,
                                 am, UT_ARCH));
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
    sprintf(buffer, "%20s, %80s", "BNN Convolution", params);
    double ops = (1.0 * on * oc * oh * ow) * (2.0 * ic * fh * fw + 1);
    ut_log(DT_I8, buffer, ops, time);

    free(input);
    free(filter);
    free(bias);
    free(output);
    free(input_ref);
    free(filter_ref);
    free(output_ref);
    free(tmp);
    free(ftm);
    return 0;
}

int main(int argc, char** argv) {
#ifdef _USE_FP16
    bnnConvolutionTest(argc, argv, DT_F16);
#endif
/*#ifdef _USE_FP32
    bnnConvolutionTest<F32>(argc, argv, DT_F32);
#endif*/
    return 0;
}
