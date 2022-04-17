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

int activationFunctionTest(U32 in,
    U32 ic,
    U32 ih,
    U32 iw,
    DataType dt,
    ActivationParamSpec activationDesc,
    const char *activationType)
{
    DataFormat df = DF_NCHWC8;
    UNI_MEMSET(activationDesc.value, 0, sizeof(activationDesc.value));

    TensorDesc dataDesc = tensor4df(dt, df, in, ic, ih, iw);
    U32 len = tensorNumElements(dataDesc);

    U8 *data = ut_input_v(len, dt, UT_INIT_RANDOM);

    Tensor dataTensor = Tensor::alloc_sized<CPUMem>(dataDesc);
    Tensor dataTensorRef = Tensor::alloc_sized<CPUMem>(dataDesc);
    UNI_MEMCPY(get_ptr_from_tensor(dataTensor, CPU_GENERAL), data, tensorNumBytes(dataDesc));
    UNI_MEMCPY(get_ptr_from_tensor(dataTensorRef, CPU_GENERAL), data, tensorNumBytes(dataDesc));

    if (UT_CHECK) {
        //check
        CHECK_STATUS(activation(dataTensor, activationDesc, dataTensor, &UT_CPU_ARCHINFO));

        // naive implement
        CHECK_STATUS(activation(dataTensorRef, activationDesc, dataTensorRef, &UT_SERIAL_ARCHINFO));

        // check
        ut_check_v(get_ptr_from_tensor(dataTensor, CPU_GENERAL),
            get_ptr_from_tensor(dataTensorRef, CPU_GENERAL), dataTensor.length(), dt, 0.01,
            __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(activation(dataTensor, activationDesc, dataTensor, &UT_CPU_ARCHINFO));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u)=(%u %u %u %u)", in, ic, ih, iw, in, ic, ih, iw);
    sprintf(buffer, "%20s, %80s", activationType, params);
    double ops = 1.0 * in * ic * ih * iw;
    ut_log(dt, buffer, ops, time / UT_LOOPS);

    free(data);

    return 0;
}

int activationTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc == 5);
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 ih = atoi(argv[3]);
    U32 iw = atoi(argv[4]);

    ActivationParamSpec activationDesc;
    //test relu
    activationDesc.mode = ACTIVATION_RELU;
    activationFunctionTest(in, ic, ih, iw, dt, activationDesc, "Activation Relu");
    //test relu6
    activationDesc.mode = ACTIVATION_RELU6;
    activationFunctionTest(in, ic, ih, iw, dt, activationDesc, "Activation Relu6");
    //test h swish
    activationDesc.mode = ACTIVATION_H_SWISH;
    activationFunctionTest(in, ic, ih, iw, dt, activationDesc, "Activation h_siwsh");
    //test h sigmod
    activationDesc.mode = ACTIVATION_H_SIGMOID;
    activationFunctionTest(in, ic, ih, iw, dt, activationDesc, "Activation h_sigmod");
    //test tanh
    activationDesc.mode = ACTIVATION_TANH;
    activationFunctionTest(in, ic, ih, iw, dt, activationDesc, "Activation tanh");
    //test gelu
    activationDesc.mode = ACTIVATION_GELU;
    activationFunctionTest(in, ic, ih, iw, dt, activationDesc, "Activation gelu");
    //test mish
    activationDesc.mode = ACTIVATION_MISH;
    activationFunctionTest(in, ic, ih, iw, dt, activationDesc, "Activation mish");
    //test sigmod
    activationDesc.mode = ACTIVATION_SIGMOID;
    activationFunctionTest(in, ic, ih, iw, dt, activationDesc, "Activation sigmod");

    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    activationTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    activationTest(argc, argv, DT_F32);
#endif
    return 0;
}
