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


int main(int argc, char **argv)
{
    CHECK_REQUIREMENT(argc == 5);
    U32 batch = atoi(argv[1]);
    U32 step  = atoi(argv[2]);
    U32 x_dim = atoi(argv[3]);
    U32 h_dim = atoi(argv[4]);

    LSTMDesc lstmDesc;
    lstmDesc.num_output = h_dim;

    TensorDesc inputDesc  = tensor3df(DT_F16, DF_MTK, batch, step, x_dim);
    TensorDesc filterDesc = tensor2df(DT_F16, DF_8NK, 4*h_dim, x_dim+h_dim);
    TensorDesc biasDesc   = tensor1d(DT_F16, h_dim*4);
    U32 outputBytes, tmpBytes, ftmBytes;
    TensorDesc outputDesc;
    CHECK_STATUS(lstm_infer_output_size(inputDesc, filterDesc, lstmDesc, &outputDesc, &outputBytes));
    CHECK_STATUS(lstm_infer_forward_tmp_bytes(inputDesc, filterDesc, outputDesc, lstmDesc, &tmpBytes, UT_ARCH));
    CHECK_STATUS(lstm_transform_filter_bytes(filterDesc, &ftmBytes, UT_ARCH));

    U32 input_len  = batch * step * x_dim;
    U32 filter_len = (x_dim + h_dim) * h_dim * 4;
    U32 bias_len   = h_dim * 4;
    U32 output_len = outputBytes / sizeof(F16);
    F16* input  = ut_input_v<F16>(input_len, UT_INIT_RANDOM);
    F16* filter = ut_input_v<F16>(filter_len, UT_INIT_RANDOM);
    F16* bias   = ut_input_v<F16>(bias_len, UT_INIT_RANDOM);
    F16* output = ut_input_v<F16>(output_len, UT_INIT_ZERO);
    F16* input_ref  = ut_input_v<F16>(input_len, UT_INIT_ZERO);
    F16* filter_ref = ut_input_v<F16>(filter_len, UT_INIT_ZERO);
    F16* bias_ref   = ut_input_v<F16>(bias_len, UT_INIT_ZERO);
    F16* output_ref = ut_input_v<F16>(output_len, UT_INIT_ZERO);
    F16 *tmp        = ut_input_v<F16>(tmpBytes/sizeof(F16), UT_INIT_ZERO);
    F16 *ftm        = ut_input_v<F16>(ftmBytes/sizeof(F16), UT_INIT_ZERO);
    memcpy(input_ref, input, input_len*sizeof(F16));
    memcpy(filter_ref, filter, filter_len*sizeof(F16));
    memcpy(bias_ref, bias, bias_len*sizeof(F16));

    TensorDesc ftmDesc;
    CHECK_STATUS(lstm_transform_filter(filterDesc, filter, &ftmDesc, ftm, x_dim, h_dim, UT_ARCH));

    if (UT_CHECK) {
        CHECK_STATUS(lstm(inputDesc, input, ftmDesc, ftm, lstmDesc, biasDesc, bias, tmpBytes, tmp, outputDesc, output, UT_ARCH));

        // naive implement
        CHECK_STATUS(lstm(inputDesc, input_ref, filterDesc, filter_ref, lstmDesc, biasDesc, bias_ref, tmpBytes, tmp, outputDesc, output_ref, CPU_GENERAL));

        // check
        ut_check_v<F16>(output, output_ref, output_len, F16(10), __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(lstm(inputDesc, input, ftmDesc, ftm, lstmDesc, biasDesc, bias, tmpBytes, tmp, outputDesc, output, UT_ARCH));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "%u (%u %u %u)=(%u %u)",
                    batch, step, x_dim, h_dim,
                    batch, h_dim);
    sprintf(buffer, "%20s, %80s", "Lstm", params);
    double hx_dim = h_dim + x_dim;
    double ops = 1.0 * batch * step * ((2.0 * hx_dim * h_dim + h_dim + h_dim) * 4 + 5.0 * h_dim);
    ut_log<F16>(buffer, ops, time);

    free(input);
    free(filter);
    free(bias);
    free(input_ref);
    free(filter_ref);
    free(bias_ref);
    free(output);
    free(output_ref);
    free(tmp);
    free(ftm);

    return 0;
}
