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

int depthwiseConvolutionTest(int argc, char *argv[], DataFormat filterDataFormat, DataType dt)
{
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

    ActivationMode dw_am = ACTIVATION_NULL;
    ActivationMode pw_am = ACTIVATION_NULL;

    U32 filterLength = 0;
    U32 biasLength = 0;
    if (filterDataFormat == DF_CHW_NC) {
        filterLength = fc*fh*fw + fn*fc;
        biasLength = ic + oc;
    }
    if (filterDataFormat == DF_NCHW) {
        oc = fc = ic;
        fn = 1;
        filterLength = fc*fh*fw;
        biasLength = ic;
    }
    if (filterLength == 0) {
        exit(1);
    }
    TensorDesc inputDesc, filterDesc, outputDesc, biasDesc;
    ConvolutionDesc convDesc;
    inputDesc = tensor4df(dt, DF_NCHWC8, in, ic, ih, iw);
    filterDesc = tensor4df(dt, filterDataFormat, oc, ic, fh, fw);
    biasDesc = tensor1d(dt, biasLength);
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
    U8 *filter = ut_input_v(filterLength, dt, UT_INIT_RANDOM);
    U8 *bias   = ut_input_v(biasLength, dt, UT_INIT_RANDOM);
    U8 *inputRef  = ut_input_v(in*ic*ih*iw, dt, UT_INIT_ZERO);
    U8 *filterRef = ut_input_v(filterLength, dt, UT_INIT_ZERO);
    U8 *biasRef   = ut_input_v(biasLength, dt, UT_INIT_ZERO);
    memcpy(inputRef,  input,  bytesOf(dt)*in*ic*ih*iw);
    memcpy(filterRef, filter, bytesOf(dt)*(filterLength));
    memcpy(biasRef,   bias,   bytesOf(dt)*(biasLength));

    // setup output, bias
    U32 outputBytes;
    CHECK_STATUS(depthwise_convolution_infer_output_size(inputDesc, filterDesc, convDesc, &outputDesc, dt, &outputBytes, UT_ARCH));
    U32 output_size = outputBytes / bytesOf(dt);
    U8 *output     = ut_input_v(output_size, dt, UT_INIT_ZERO);
    U8 *outputRef = ut_input_v(output_size, dt, UT_INIT_ZERO);

    // setup alg
    ConvolutionPolicy policy = CONVOLUTION_FASTEST;
    DepthwiseConvolutionForwardAlgorithm alg = DEPTHWISE_CONVOLUTION_ALGORITHM_NULL;
    CHECK_STATUS(depthwise_convolution_infer_forward_algorithm(inputDesc, filterDesc, outputDesc, convDesc, policy, &alg, dt, dw_am, pw_am, UT_ARCH));

    // setup tmp
    U32 tmpBytes;
    CHECK_STATUS(depthwise_convolution_infer_forward_tmp_bytes(inputDesc, filterDesc, outputDesc, convDesc, alg, &tmpBytes, UT_ARCH));
    U8 *tmp     = ut_input_v(tmpBytes/bytesOf(dt), dt, UT_INIT_ZERO);

    // setup filter trans
    U32 ftmBytes;
    CHECK_STATUS(depthwise_convolution_transform_filter_bytes(filterDesc, alg, &ftmBytes, UT_ARCH));
    U8 *ftm     = ut_input_v(ftmBytes/bytesOf(dt), dt, UT_INIT_ZERO);
    // trans filter
    TensorDesc ftmDesc;
    CHECK_STATUS(depthwise_convolution_transform_filter(filterDesc, filter, alg, &ftmDesc, ftm, UT_ARCH));

    if (UT_CHECK) {
        CHECK_STATUS(depthwise_convolution(inputDesc, input,
                                           ftmDesc, ftm,
                                           convDesc, alg,
                                           biasDesc, bias,
                                           tmpBytes, tmp,
                                           outputDesc, output,
                                           dw_am, pw_am,
                                           UT_ARCH));

        // naive implement
        CHECK_STATUS(depthwise_convolution(inputDesc, inputRef,
                                           filterDesc, filterRef,
                                           convDesc, alg,
                                           biasDesc, biasRef,
                                           tmpBytes, tmp,
                                           outputDesc, outputRef,
                                           dw_am, pw_am,
                                           CPU_GENERAL));

        // check
        ut_check_v(output, outputRef, output_size, dt, 0.1, __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for(int iter = 0; iter < UT_LOOPS; iter++){
        CHECK_STATUS(depthwise_convolution(inputDesc, input,
                                           ftmDesc, ftm,
                                           convDesc, alg,
                                           biasDesc, bias,
                                           tmpBytes, tmp,
                                           outputDesc, output,
                                           dw_am, pw_am,
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
    double ops = 0;
    if (filterDataFormat == DF_CHW_NC) {
        sprintf(buffer, "%20s, %80s", "DepthwisePointwise", params);
        ops = 2.0 * in * ic * ih * iw * fh * fw + in * ic * oh * ow +
                 2.0 * on * oc * oh * ow * ic + on * oc * oh * ow;
    }
    if (filterDataFormat == DF_NCHW) {
        sprintf(buffer, "%20s, %80s", "DepthwiseConvolution", params);
        ops = 2.0 * in * ic * ih * iw * fh * fw + in * ic * oh * ow;
    }
    ut_log(dt, buffer, ops, time);

    free(input);
    free(filter);
    free(bias);
    free(output);
    free(inputRef);
    free(filterRef);
    free(biasRef);
    free(outputRef);
    free(tmp);
    free(ftm);

    return 0;
}

int main(int argc, char *argv[])
{
#ifdef _USE_FP16
    depthwiseConvolutionTest(argc, argv, DF_CHW_NC, DT_F16);
    depthwiseConvolutionTest(argc, argv, DF_NCHW, DT_F16);
#endif
#ifdef _USE_FP32
    depthwiseConvolutionTest(argc, argv, DF_CHW_NC, DT_F32);
    depthwiseConvolutionTest(argc, argv, DF_NCHW, DT_F32);
#endif
    return 0;
}
