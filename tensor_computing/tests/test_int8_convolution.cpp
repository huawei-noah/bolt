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
#include "utils.h"

int main(int argc, char* argv[]){
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

    Arch arch = ARM_A55;
    DataType odt = DT_F16;
    ActivationMode am = ACTIVATION_RELU;

    TensorDesc inputDesc, filterDesc, outputDesc, biasDesc;
    ConvolutionDesc convDesc;
    if (ic == 3 || ic == 1) {
        printf("Now never quantize the first layer\n");
        return 0;
    } else {
        DataType dt = DT_F16;
        DataType qdt = DT_I8;
        TensorDesc inputDesc_ref = tensor4df(dt, DF_NCHWC8, in, ic, ih, iw);
        filterDesc = tensor4df(dt, DF_NCHW, oc, ic, fh, fw);
        biasDesc = tensor1d(odt, oc);
        convDesc.stride = stride;
        convDesc.padding = padding;
        convDesc.dilatedRate = 1;

        // setup input, filter, bias
        F16 *input_ref  = ut_input_v<F16>(in*ic*ih*iw, UT_INIT_RANDOM);
        F16 *filter = ut_input_v<F16>(fn*fc*fh*fw, UT_INIT_RANDOM);
        F16 *bias   = ut_input_v<F16>(oc, UT_INIT_RANDOM);

        INT8 *input  = ut_input_v<INT8>(in*ic*ih*iw, UT_INIT_ZERO);
        F16 scale_i;
        quantize_tensor(inputDesc_ref, input_ref, &inputDesc, input, &scale_i);

        F16 *filter_ref = ut_input_v<F16>(fn*fc*fh*fw, UT_INIT_ZERO);
        memcpy(filter_ref, filter, bytesOf(dt)*fn*fc*fh*fw);

        // setup output, bias
        U32 outputBytes;
        CHECK_STATUS(convolution_infer_output_size(inputDesc, filterDesc, convDesc, &outputDesc, qdt, &outputBytes));
        TensorDesc outputDesc_ref = outputDesc;
        outputDesc_ref.dt = DT_F16;
        U32 output_size = outputBytes / bytesOf(qdt);
        INT8 *output     = ut_input_v<INT8>(output_size, UT_INIT_ZERO);
        F16 *output_ref = ut_input_v<F16>(output_size, UT_INIT_ZERO);

        // setup alg
        ConvolutionPolicy policy = CONVOLUTION_FASTEST;
        ConvolutionForwardAlgorithm alg;
        CHECK_STATUS(convolution_infer_forward_algorithm(inputDesc, filterDesc, outputDesc, convDesc, policy, &alg, DT_I8, arch));

        F16 *scales;

        // setup filter trans
        U32 ftBytes;

        TensorDesc ftmDesc;
        INT8 *ftm;

        switch (alg) {
            case CONVOLUTION_ALGORITHM_WINOGRAD: {
                CHECK_STATUS(convolution_transform_filter_bytes(filterDesc, alg, &ftBytes, arch));

                TensorDesc tFilterDesc;
                F16 *tFilter = ut_input_v<F16>(ftBytes/sizeof(F16), UT_INIT_ZERO);

                filterDesc.dt = DT_F16_8Q;  // To label as int8
                CHECK_STATUS(convolution_transform_filter(filterDesc, filter, alg, &tFilterDesc, tFilter, arch));
                filterDesc.dt = DT_F16;

                ftm     = ut_input_v<INT8>(fn*fc*6*6, UT_INIT_ZERO);

                scales = ut_input_v<F16>(38, UT_INIT_ZERO);  // 1 for input, 1 for output and 36 for filter
                CHECK_STATUS(quantize_tensor(tFilterDesc, tFilter, &ftmDesc, ftm, scales+2));

                free(tFilter);
                break;
            }
            default: {
                TensorDesc qFilterDesc;
                INT8 *qFilter = ut_input_v<INT8>(fn*fc*fh*fw, UT_INIT_ZERO);
                scales = ut_input_v<F16>(3, UT_INIT_ZERO);
                CHECK_STATUS(quantize_tensor(filterDesc, filter, &qFilterDesc, qFilter, scales+2));

                CHECK_STATUS(convolution_transform_filter_bytes(qFilterDesc, alg, &ftBytes, arch));
                
                ftm     = ut_input_v<INT8>(ftBytes/sizeof(INT8), UT_INIT_ZERO);
                // trans filter
                CHECK_STATUS(convolution_transform_filter(qFilterDesc, qFilter, alg, &ftmDesc, ftm, arch));

                free(qFilter);
                break;
            }
        }

        scales[0] = scale_i;

        // setup tmp
        U32 tmpBytes;
        CHECK_STATUS(convolution_infer_forward_tmp_bytes(inputDesc, ftmDesc, outputDesc, convDesc, alg, &tmpBytes, arch));
        INT8 *tmp     = ut_input_v<INT8>(tmpBytes/sizeof(INT8), UT_INIT_ZERO);
        
        if (UT_CHECK) {
            CHECK_STATUS(convolution(inputDesc, input,
                                 ftmDesc, ftm,
                                 convDesc, alg,
                                 biasDesc, scales,
                                 biasDesc, bias,
                                 tmpBytes, tmp,
                                 outputDesc, output,
                                 am, arch));

            // naive implement
            CHECK_STATUS(convolution(inputDesc_ref, input_ref,
                                 filterDesc, filter_ref,
                                 convDesc, alg,
                                 biasDesc, nullptr,
                                 biasDesc, bias,
                                 tmpBytes, tmp,
                                 outputDesc_ref, output_ref,
                                 am, CPU_GENERAL));
            
            F16 *out_d = ut_input_v<F16>(output_size, UT_INIT_ZERO);
            for (U32 i = 0; i < output_size; i++) {
                out_d[i] = output[i] / scales[1];
            }
            ut_check_v<F16>(out_d, output_ref, output_size, (F16)8, __FILE__, __LINE__);
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
                                 am, arch));
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
        ut_log<INT8>(buffer, ops, time);

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
