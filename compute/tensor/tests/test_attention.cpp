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

int attentionTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc == 5);
    AttentionParamSpec p;
    U32 batch = atoi(argv[1]);
    p.num_heads = atoi(argv[2]);
    p.from_sequence_length = atoi(argv[3]);
    p.to_sequence_length = atoi(argv[4]);
    ArchInfo archInfo;
    archInfo.arch = UT_ARCH;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;

    DataFormat df = DF_NORMAL;
    TensorDesc inDesc = tensor2df(dt, df, batch, p.to_sequence_length);
    U32 inputLength = tensorNumElements(inDesc);
    U8 *input = ut_input_v(inputLength, dt, UT_INIT_ZERO);
    Tensor inputTensor = Tensor::alloc_sized<CPUMem>(inDesc);

    Tensor outputTensor;
    CHECK_STATUS(attention_infer_output_size(&inputTensor, p, &outputTensor));
    outputTensor.alloc();
    Tensor outputTensorRef = Tensor::alloc_sized<CPUMem>(outputTensor.get_desc());
    U32 outputLength = outputTensor.length();
    for (U32 i = 0; i < batch; i++) {
        U32 threshold = p.to_sequence_length / 2 + i;
        for (U32 j = 0; j < p.to_sequence_length; j++) {
            if (j < threshold) {
                switch (dt) {
#ifdef _USE_FP32
                    case DT_F32:
                        ((F32 *)input)[i * p.to_sequence_length + j] = 1;
                        break;
#endif
#ifdef _USE_FP16
                    case DT_F16:
                        ((F16 *)input)[i * p.to_sequence_length + j] = 1;
                        break;
#endif
                    default:
                        break;
                }
            }
        }
    }

    memcpy(get_ptr_from_tensor(inputTensor, UT_ARCH), input, tensorNumBytes(inDesc));

    if (UT_CHECK) {
        CHECK_STATUS(attention(inputTensor, outputTensor, &archInfo));

        // naive implement
        CHECK_STATUS(attention(inputTensor, outputTensorRef, &archInfo_org));

        // check
        ut_check_v(get_ptr_from_tensor(outputTensor, UT_ARCH),
            get_ptr_from_tensor(outputTensorRef, UT_ARCH), outputLength, dt, 0, __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(attention(inputTensor, outputTensor, &archInfo));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u)=(%u %u %u %u)", batch, p.from_sequence_length, batch, p.num_heads,
        p.from_sequence_length, p.to_sequence_length);
    sprintf(buffer, "%20s, %80s", "Attention", params);
    double ops = 3.0 * outputLength;
    ut_log(dt, buffer, ops, time / UT_LOOPS);

    free(input);

    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    attentionTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    attentionTest(argc, argv, DT_F32);
#endif
    return 0;
}
