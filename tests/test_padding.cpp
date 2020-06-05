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

int paddingTest(int argc, char **argv, DataType dt) {
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

    //  output dim
    U32 on = in + n_fir + n_sec;
    U32 oc = ic + c_fir + c_sec;
    U32 oh = ih + h_fir + h_sec;
    U32 ow = iw + w_fir + w_sec;

    PadDesc padDesc;
    
    padDesc.top = h_fir;
    padDesc.bottom = h_sec;
    padDesc.left = w_fir; 
    padDesc.right = w_sec;
    padDesc.constant_value = 0.0;
    // padDesc.pad_mode = Pad_Constant;    
    // padDesc.pad_mode = Pad_Reflect;    //limitation: the h_fir and the h_sec should lower than 0
    padDesc.pad_mode = Pad_Edge;   

    TensorDesc input_desc = tensor4df(dt, DF_NCHW, in, ic, ih, iw);
    TensorDesc output_desc;
    CHECK_STATUS(padding_infer_output_size(input_desc, padDesc, &output_desc));
    U32 input_len = tensorNumElements(input_desc);
    U32 output_len = tensorNumElements(output_desc);
    U8* input = (U8*)malloc(input_len * sizeof(dt));
    F16* input_assign = (F16*)input;
    for (int i=0; i<input_len; i++) {
        input_assign[i] = (F16)(i + 1); 
    }
    U8* output = (U8*)malloc(output_len * sizeof(dt));

    if (UT_CHECK) {
        CHECK_STATUS(padding(input_desc, input, padDesc, output_desc, output, CPU_GENERAL));
    }
    F16* output_ptr = (F16*)output;
    for (int i=0; i<output_len; i++) {
        std::cout << output_ptr[i] << " ";
        if (((i + 1)%(iw + w_fir + w_sec)) == 0) {
            std::cout << std::endl;
        }
    }
    free(input);
    free(output);

    return 0;
}


int main(int argc, char** argv) {
#ifdef _USE_FP16
    std::cout << "testing Fp16" << std::endl;
    paddingTest(argc, argv, DT_F16);
#endif

/*
#ifdef _USE_FP32
    std::cout << "testing Fp32" << std::endl;
    paddingTest(argc, argv, DT_F32);
#endif
*/
    return 0;
}
