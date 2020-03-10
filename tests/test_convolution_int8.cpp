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

#ifdef _USE_INT8
int int8ConvolutionTest(int argc, char* argv[], DataType dt, DataType filterDataType) {
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

    ActivationMode am = ACTIVATION_RELU;

    TensorDesc inputDesc, filterDesc, outputDesc, biasDesc;
    ConvolutionDesc convDesc;
    if (ic == 3 || ic == 1) {
        printf("[WARN] can not quantize the first layer\n");
        return 0;
    } else {
        DataType qdt = DT_I8;
        TensorDesc inputDesc_ref = tensor4df(dt, DF_NCHWC8, in, ic, ih, iw);
        filterDesc = tensor4df(dt, DF_NCHW, oc, ic, fh, fw);
        biasDesc = tensor1d(dt, oc);
        convDesc.stride_h = stride;
        convDesc.stride_w = stride;
        convDesc.padding_top = padding;
        convDesc.padding_bottom = padding;
        convDesc.padding_left = padding;
        convDesc.padding_right = padding;
        convDesc.dilatedRate_h = 1;
        convDesc.dilatedRate_w = 1;

        // setup input, filter, bias
        U8 *input_ref  = ut_input_v(in*ic*ih*iw, dt, UT_INIT_RANDOM);
        U8 *filter = ut_input_v(fn*fc*fh*fw, dt, UT_INIT_RANDOM);
        U8 *bias   = ut_input_v(oc, dt, UT_INIT_RANDOM);

        INT8 *input = (INT8*)ut_input_v(in*ic*ih*iw, DT_I8, UT_INIT_ZERO);
        F16 scale_i;
        quantize_tensor(inputDesc_ref, input_ref, &inputDesc, input, &scale_i);

        U8 *filter_ref = ut_input_v(fn*fc*fh*fw, dt, UT_INIT_ZERO);
        memcpy(filter_ref, filter, bytesOf(dt)*fn*fc*fh*fw);

        // setup output, bias
        U32 outputBytes;
        CHECK_STATUS(convolution_infer_output_size(inputDesc, filterDesc, convDesc, &outputDesc, qdt, &outputBytes, UT_ARCH));
        TensorDesc outputDesc_ref = outputDesc;
        outputDesc_ref.dt = dt;
        U32 output_size = outputBytes / bytesOf(qdt);
        INT8 *output   = (INT8*)ut_input_v(output_size, DT_I8, UT_INIT_ZERO);
        U8 *output_ref = ut_input_v(output_size, dt, UT_INIT_ZERO);

        // setup alg
        ConvolutionPolicy policy = CONVOLUTION_FASTEST;
        ConvolutionForwardAlgorithm alg = CONVOLUTION_ALGORITHM_NULL;
        CHECK_STATUS(convolution_infer_forward_algorithm(inputDesc, filterDesc, outputDesc, convDesc, policy, &alg, DT_I8, am, UT_ARCH));

        F16 *scales;

        // setup filter trans
        U32 ftBytes;

        TensorDesc ftmDesc;
        INT8 *ftm;

        switch (alg) {
            case CONVOLUTION_ALGORITHM_WINOGRAD: {
                CHECK_STATUS(convolution_transform_filter_bytes(filterDesc, alg, &ftBytes, UT_ARCH));

                TensorDesc tFilterDesc;
                U8 *tFilter = ut_input_v(ftBytes/bytesOf(dt), dt, UT_INIT_ZERO);

                filterDesc.dt = filterDataType;  // To label as int8
                CHECK_STATUS(convolution_transform_filter(filterDesc, filter, alg, &tFilterDesc, tFilter, UT_ARCH));
                filterDesc.dt = dt;

                ftm     = (INT8*)ut_input_v(fn*fc*6*6, DT_I8, UT_INIT_ZERO);

                scales = (F16*)ut_input_v(38, DT_F16, UT_INIT_ZERO);  // 1 for input, 1 for output and 36 for filter
                CHECK_STATUS(quantize_tensor(tFilterDesc, tFilter, &ftmDesc, ftm, scales+2));

                free(tFilter);
                break;
            }
            default: {
                TensorDesc qFilterDesc;
                INT8 *qFilter = (INT8*)ut_input_v(fn*fc*fh*fw, DT_I8, UT_INIT_ZERO);
                scales = (F16*)ut_input_v(3, DT_F16, UT_INIT_ZERO);
                CHECK_STATUS(quantize_tensor(filterDesc, filter, &qFilterDesc, qFilter, scales+2));

                CHECK_STATUS(convolution_transform_filter_bytes(qFilterDesc, alg, &ftBytes, UT_ARCH));
                
                ftm     = (INT8*)ut_input_v(ftBytes/sizeof(INT8), DT_I8, UT_INIT_ZERO);
                // trans filter
                CHECK_STATUS(convolution_transform_filter(qFilterDesc, qFilter, alg, &ftmDesc, ftm, UT_ARCH));

                free(qFilter);
                break;
            }
        }

        scales[0] = scale_i;

        // setup tmp
        U32 tmpBytes;
        CHECK_STATUS(convolution_infer_forward_tmp_bytes(inputDesc, ftmDesc, outputDesc, convDesc, alg, &tmpBytes, UT_ARCH));
        INT8 *tmp     = (INT8*)ut_input_v(tmpBytes/sizeof(INT8), DT_I8, UT_INIT_ZERO);
        
        if (UT_CHECK) {
            CHECK_STATUS(convolution(inputDesc, input,
                                 ftmDesc, ftm,
                                 convDesc, alg,
                                 biasDesc, scales,
                                 biasDesc, bias,
                                 tmpBytes, tmp,
                                 outputDesc, output,
                                 am, UT_ARCH));

            // naive implement
            CHECK_STATUS(convolution(inputDesc_ref, input_ref,
                                 filterDesc, filter_ref,
                                 convDesc, alg,
                                 biasDesc, nullptr,
                                 biasDesc, bias,
                                 tmpBytes, tmp,
                                 outputDesc_ref, output_ref,
                                 am, CPU_GENERAL));
            
            U8 *out_d = ut_input_v(output_size, dt, UT_INIT_ZERO);
            for (U32 i = 0; i < output_size; i++) {
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
            ut_check_v(out_d, output_ref, output_size, dt, 8, __FILE__, __LINE__);
            free(out_d);
        }

        // benchmark
        double time_start = ut_time_ms();
        for (int iter = 0; iter < UT_LOOPS; iter++){
            CHECK_STATUS(convolution(inputDesc, input,
                                 ftmDesc, ftm,
                                 convDesc, alg,
                                 biasDesc, scales,
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
        sprintf(buffer, "%20s, %80s", "Convolution", params);
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
        free(scales);
    }
    return 0;
}
#endif

int main(int argc, char** argv) {
#ifdef _USE_INT8
    int8ConvolutionTest(argc, argv, DT_F16, DT_F16_8Q);
#endif
    return 0;
}
