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

int paddingTest(int argc, char **argv, DataType dt)
{
    // input dim
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 ih = atoi(argv[3]);
    U32 iw = atoi(argv[4]);

    // padding info
    U32 n_fir = atoi(argv[5]);
    U32 c_fir = atoi(argv[6]);
    U32 h_fir = atoi(argv[7]);
    U32 w_fir = atoi(argv[8]);
    U32 n_sec = atoi(argv[9]);
    U32 c_sec = atoi(argv[10]);
    U32 h_sec = atoi(argv[11]);
    U32 w_sec = atoi(argv[12]);
    U32 mode = atoi(argv[13]);
    CHECK_REQUIREMENT(n_fir == 0 && n_sec == 0 && c_fir == 0 && c_sec == 0);

    CHECK_REQUIREMENT(0 == n_fir);
    CHECK_REQUIREMENT(0 == n_sec);
    CHECK_REQUIREMENT(0 == c_fir);
    CHECK_REQUIREMENT(0 == c_sec);

    ArchInfo archInfo;
    archInfo.arch = UT_ARCH;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;
    PadParamSpec padParamSpec;

    padParamSpec.top = h_fir;
    padParamSpec.bottom = h_sec;
    padParamSpec.left = w_fir;
    padParamSpec.right = w_sec;
    padParamSpec.constant_value = 0.0;
    switch (mode) {
        case 0: {
            padParamSpec.pad_mode = Pad_Constant;
            break;
        }
        case 1: {
            padParamSpec.pad_mode = Pad_Edge;
            break;
        }
        case 2: {
            // limitation: the h_fir and the h_sec should lower than 0
            padParamSpec.pad_mode = Pad_Reflect;
            break;
        }
        case 3: {
            padParamSpec.pad_mode = Pad_Symmetric;
            break;
        }
        default: {
            UNI_ERROR_LOG("unknown paddding mode %d\n", mode);
            break;
        }
    }

    Tensor inputTensor;
    TensorDesc inputDesc = tensor4df(dt, DF_NCHW, in, ic, ih, iw);
    inputTensor.resize(inputDesc);
    inputTensor.alloc();
    U32 input_len = tensorNumElements(inputDesc);
    U8 *input = ut_input_v(input_len, dt, UT_INIT_RANDOM);
    memcpy(get_ptr_from_tensor(inputTensor, UT_ARCH), input, tensorNumBytes(inputDesc));

    // set output
    Tensor outputTensor, outputTensorRef;
    CHECK_STATUS(
        padding_infer_output_size(&inputTensor, padParamSpec, &outputTensor, &archInfo_org));
    outputTensor.alloc();
    TensorDesc outputDesc_ref = outputTensor.get_desc();
    outputTensorRef.resize(outputDesc_ref);
    outputTensorRef.alloc();
    U32 output_len = outputTensor.length();

    if (UT_CHECK) {
        CHECK_STATUS(padding(inputTensor, padParamSpec, outputTensor, &archInfo));

        CHECK_STATUS(padding(inputTensor, padParamSpec, outputTensorRef, &archInfo_org));

        // check
        ut_check_a(get_ptr_from_tensor(outputTensor, UT_ARCH),
            get_ptr_from_tensor(outputTensorRef, UT_ARCH), output_len, dt);
    }

    free(input);
    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    paddingTest(argc, argv, DT_F16);
#endif
    return 0;
}
