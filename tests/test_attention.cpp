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

int attentionTest(int argc, char** argv, DataType dt) {
    CHECK_REQUIREMENT(argc == 5);
    U32 batch = atoi(argv[1]);
    U32 numHeads = atoi(argv[2]);
    U32 fromSequenceLength = atoi(argv[3]);
    U32 toSequenceLength = atoi(argv[4]);

    DataFormat df = DF_NORMAL;
    TensorDesc inputDesc = tensor2df(dt, df, batch, toSequenceLength); 
    TensorDesc outputDesc;
    CHECK_STATUS(attention_infer_output_size(inputDesc, numHeads, fromSequenceLength, toSequenceLength, &outputDesc));
    U32 inputLength  = tensorNumElements(inputDesc);
    U32 outputLength = tensorNumElements(outputDesc);

    U8* input  = ut_input_v(inputLength, dt, UT_INIT_ZERO);
    for (U32 i = 0; i < batch; i++) {
        U32 threshold = toSequenceLength / 2 + i;
        for (U32 j = 0; j < toSequenceLength; j++) {
            if (j < threshold) {
                switch (dt) {
#ifdef _USE_FP32
                    case DT_F32:
                        ((F32*)input)[i * toSequenceLength + j] = 1;
                        break;
#endif
#ifdef _USE_FP16
                    case DT_F16:
                        ((F16*)input)[i * toSequenceLength + j] = 1;
                        break;
#endif
                    default:
                        break;
                }
            }
        }
    }
    U8* output    = ut_input_v(outputLength, dt, UT_INIT_ZERO);
    U8* outputRef = ut_input_v(outputLength, dt, UT_INIT_ZERO);
    if(UT_CHECK) {
        CHECK_STATUS(attention(inputDesc, input, outputDesc, output, UT_ARCH));

        // naive implement
        CHECK_STATUS(attention(inputDesc, input, outputDesc, outputRef, CPU_GENERAL));

        // check
        ut_check_v(output, outputRef, outputLength, dt, 0, __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for(int iter=0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(attention(inputDesc, input, outputDesc, output, UT_ARCH));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u)=(%u %u %u %u)",
                    batch, fromSequenceLength,
                    batch, numHeads, fromSequenceLength, toSequenceLength);
    sprintf(buffer, "%20s, %80s", "Attention", params);
    double ops = 3.0 * outputLength;
    ut_log(dt, buffer, ops, time/UT_LOOPS);

    free(input);
    free(output);
    free(outputRef);

    return 0;
}

int main(int argc, char** argv) {
#ifdef _USE_FP16
    attentionTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    attentionTest(argc, argv, DT_F32);
#endif
    return 0;
}
