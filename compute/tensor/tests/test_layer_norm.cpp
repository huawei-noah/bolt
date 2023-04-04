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

int layerNormTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc == 6);
    U32 alpha = atoi(argv[1]);
    U32 beta = atoi(argv[2]);
    U32 ic = atoi(argv[3]);
    U32 ih = atoi(argv[4]);
    U32 iw = atoi(argv[5]);
    LayerNormParamSpec p;
    p.axis = -1;

    DataFormat df = DF_MTK;
    Tensor inputTensor;
    TensorDesc inputDesc = tensor3df(dt, df, ic, ih, iw);
    inputTensor.resize(inputDesc);
    inputTensor.alloc();
    U32 input_len = tensorNumElements(inputDesc);
    U8 *input = ut_input_v(input_len, dt, UT_INIT_RANDOM);
    UNI_MEMCPY(get_ptr_from_tensor(inputTensor, CPU_GENERAL), input, tensorNumBytes(inputDesc));

    // set output
    Tensor outputTensor, outputTensorRef;
    CHECK_STATUS(layer_norm_infer_output_size(&inputTensor, p, &outputTensor, &UT_CPU_ARCHINFO));
    outputTensor.alloc();
    TensorDesc outputDesc_ref = outputTensor.get_desc();
    outputTensorRef.resize(outputDesc_ref);
    outputTensorRef.alloc();
    U32 output_len = outputTensor.length();
    CHECK_REQUIREMENT(input_len == ic * ih * iw && output_len == ic * ih * iw);

    U32 alpha_list[ic];
    U32 beta_list[ic];
    for (int i = 0; i < (int)ic; i++) {
        alpha_list[i] = alpha;
        beta_list[i] = beta;
    }
    Tensor alphaTensor, betaTensor, tmpTensor;
    TensorDesc alphaDesc, betaDesc;
    alphaDesc = tensor1d(dt, ic);
    betaDesc = tensor1d(dt, ic);
    alphaTensor.resize(alphaDesc);
    betaTensor.resize(betaDesc);
    alphaTensor.alloc();
    betaTensor.alloc();
    UNI_MEMCPY(get_ptr_from_tensor(alphaTensor, CPU_GENERAL), alpha_list, tensorNumBytes(alphaDesc));
    UNI_MEMCPY(get_ptr_from_tensor(betaTensor, CPU_GENERAL), beta_list, tensorNumBytes(betaDesc));

    if (UT_CHECK) {
        CHECK_STATUS(layer_norm(
            inputTensor, p, alphaTensor, betaTensor, tmpTensor, outputTensor, &UT_CPU_ARCHINFO));

        // naive implement
        CHECK_STATUS(layer_norm(inputTensor, p, alphaTensor, betaTensor, tmpTensor,
            outputTensorRef, &UT_SERIAL_ARCHINFO));

        // check
        ut_check_v(get_ptr_from_tensor(outputTensor, CPU_GENERAL),
            get_ptr_from_tensor(outputTensorRef, CPU_GENERAL), output_len, dt, 0.000001);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(layer_norm(
            inputTensor, p, alphaTensor, betaTensor, tmpTensor, outputTensor, &UT_CPU_ARCHINFO));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u)=(%u %u %u)", ic, ih, iw, ic, ih, iw);
    sprintf(buffer, "%20s, %80s", "Normalization", params);
    double ops = input_len;
    ut_log(dt, buffer, ops, time);

    free(input);

    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    layerNormTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    layerNormTest(argc, argv, DT_F32);
#endif
    return 0;
}
