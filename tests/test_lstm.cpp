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

int lstmTest(int argc, char **argv, DataType dt) {
    CHECK_REQUIREMENT(argc == 5);
    U32 batch = atoi(argv[1]);
    U32 step  = atoi(argv[2]);
    U32 xDim = atoi(argv[3]);
    U32 hDim = atoi(argv[4]);

    LSTMDesc lstmDesc;
    lstmDesc.numOutput = hDim;
    lstmDesc.forgetBias = 1.0;
    lstmDesc.activationMode = ACTIVATION_TANH;

    TensorDesc inputDesc  = tensor3df(dt, DF_MTK, batch, step, xDim);
    TensorDesc filterDesc = tensor2df(dt, DF_NK, 4*hDim, xDim+hDim);
    TensorDesc biasDesc   = tensor1d(dt, hDim*4);
    U32 outputBytes, tmpBytes, ftmBytes;
    TensorDesc outputDesc;
    CHECK_STATUS(lstm_infer_output_size(inputDesc, filterDesc, lstmDesc, &outputDesc, &outputBytes));
    CHECK_STATUS(lstm_infer_forward_tmp_bytes(inputDesc, filterDesc, outputDesc, lstmDesc, &tmpBytes, UT_ARCH));
    CHECK_STATUS(lstm_transform_filter_bytes(filterDesc, &ftmBytes, UT_ARCH));

    U32 inputLength  = batch * step * xDim;
    U32 filterLength = (xDim + hDim) * hDim * 4;
    U32 biasLength   = hDim * 4;
    U32 outputLength = outputBytes / bytesOf(dt);
    U8* input  = ut_input_v(inputLength, dt, UT_INIT_RANDOM);
    U8* filter = ut_input_v(filterLength, dt, UT_INIT_RANDOM);
    U8* bias   = ut_input_v(biasLength, dt, UT_INIT_RANDOM);
    U8* output = ut_input_v(outputLength, dt, UT_INIT_ZERO);
    U8* outputRef = ut_input_v(outputLength, dt, UT_INIT_ZERO);
    U8* tmp       = ut_input_v(tmpBytes/bytesOf(dt), dt, UT_INIT_ZERO);
    U8* ftm       = ut_input_v(ftmBytes/bytesOf(dt), dt, UT_INIT_ZERO);

    TensorDesc ftmDesc;
    CHECK_STATUS(lstm_transform_filter(filterDesc, filter, &ftmDesc, ftm, xDim, hDim, UT_ARCH));

    if (UT_CHECK) {
        CHECK_STATUS(lstm(inputDesc, input, ftmDesc, ftm, biasDesc, bias, tmpBytes, tmp, lstmDesc, outputDesc, output, UT_ARCH));

        // naive implement
        CHECK_STATUS(lstm(inputDesc, input, ftmDesc, ftm, biasDesc, bias, tmpBytes, tmp, lstmDesc, outputDesc, outputRef, CPU_GENERAL));

        // check
        ut_check_v(output, outputRef, outputLength, dt, 10, __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(lstm(inputDesc, input, ftmDesc, ftm, biasDesc, bias, tmpBytes, tmp, lstmDesc, outputDesc, output, UT_ARCH));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "%u (%u %u %u)=(%u %u)",
                    batch, step, xDim, hDim,
                    batch, hDim);
    sprintf(buffer, "%20s, %80s", "Lstm", params);
    double hxDim = hDim + xDim;
    double ops = 1.0 * batch * step * ((2.0 * hxDim * hDim + hDim + hDim) * 4 + 5.0 * hDim);
    ut_log(dt, buffer, ops, time);

    free(input);
    free(filter);
    free(bias);
    free(output);
    free(outputRef);
    free(tmp);
    free(ftm);

    return 0;
}


int main(int argc, char** argv) {
#ifdef _USE_FP16
    lstmTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    lstmTest(argc, argv, DT_F32);
#endif
    return 0;
}
