// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "cpu/general/tensor_computing_general.h"

EE eltwise(std::vector<void*>input, U32 num, U32 len, void *output, EltwiseMode eltwiseMode){
    F16 value_s, tmp_s;
    for (U32 i = 0; i < len; i++) {
        tmp_s = *((F16*)(input[0]) + i);

        for (U32 j = 1; j < num; j++) {
            value_s = *((F16*)(input[j]) + i);
            switch (eltwiseMode) {
                case ELTWISE_SUM:
                    tmp_s = value_s + tmp_s;
                    break;
                case ELTWISE_MAX:
                    tmp_s = (value_s > tmp_s) ? value_s : tmp_s;
                    break;
                case ELTWISE_PROD:
                    tmp_s *= value_s;
                    break;
                default:
                    return NOT_SUPPORTED;
            }
        }
        *((F16*)output + i) = tmp_s;
    }
    return SUCCESS;
}

EE eltwise_general(std::vector<TensorDesc> inputDesc, std::vector<void*> input,
    TensorDesc outputDesc, void* output, EltwiseMode eltwiseMode)
{
    int num = inputDesc.size();
    if(num <= 1) return NOT_MATCH;

    U32 len = tensorNumElements(outputDesc);

    EE ret = SUCCESS;
    switch (outputDesc.dt) {
        case DT_F16: {
            ret = eltwise(input, num, len, output, eltwiseMode);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
