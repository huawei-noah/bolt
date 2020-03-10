// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <vector>
#include "cpu/arm/tensor_computing_arm.h"
#ifdef _USE_FP32
#include "cpu/arm/fp32/tensor_computing_fp32.h"
#endif
#ifdef _USE_FP16
#include "cpu/arm/fp16/tensor_computing_fp16.h"
#endif

EE eltwise_arm(std::vector<TensorDesc> inputDesc, std::vector<void*> input,
               TensorDesc outputDesc, void* output, EltwiseMode eltwiseMode) {
    U32 num = inputDesc.size();
    if(num <= 1) return NOT_MATCH;
    U32 batch = outputDesc.dims[outputDesc.nDims - 1];
    std::vector<U32> batchs(num, 1);
    for (U32 i = 0; i < num; i++) {
        if (inputDesc[i].dims[inputDesc[i].nDims - 1] != batch)
            batchs[i] = 0;
    }

    U32 arrayDimMin = 0;
    for (U32 i = 1; i < num; i++) {
        if (inputDesc[i].nDims < inputDesc[arrayDimMin].nDims)
            arrayDimMin = i;
    }
    U32 sameDim = 0;
    for (U32 i = 0; i < inputDesc[arrayDimMin].nDims; i++) {
        bool various = false;
        for (U32 j = 1; j < num; j++) {
            if (inputDesc[j].dims[i] != inputDesc[0].dims[i])
                various = true;
        }
        if (various)
            break;
        else
            sameDim++;
    }
    U32 loopInner = 1;
    for (U32 i = 0; i < sameDim; i++) {
        loopInner *= inputDesc[0].dims[i];
    }
    U32 len = tensorNumElements(outputDesc);
    U32 loopOuter = len / batch / loopInner;
    std::vector<U32> loopOuters(num);
    for (U32 i = 0; i < num; i++) {
        if (batchs[i] != 0)
            loopOuters[i] = tensorNumElements(inputDesc[i]) / batch / loopInner;
        else
            loopOuters[i] = tensorNumElements(inputDesc[i]) / loopInner;
    }

    EE ret = SUCCESS;
    for (U32 i = 0; i < batch; i++) {
        for (U32 j = 0; j < loopOuter; j++) {
            std::vector<void*> currentInput(num, nullptr);
            void *currentOutput = (U8*)output + ((i * loopOuter + j) * loopInner) * bytesOf(outputDesc.dt);
            for (U32 k = 0; k < num; k++) {
                U32 curJ = 0;
                if (j < loopOuters[k])
                    curJ = j;
                currentInput[k] = (U8*)input[k] + ((i * batchs[k] * loopOuters[k] + curJ) * loopInner) * bytesOf(inputDesc[k].dt);
            }
            switch (outputDesc.dt) {
#ifdef _USE_FP32
                case DT_F32: {
                    ret = eltwise_fp32(currentInput, num, loopInner, currentOutput, eltwiseMode);
                    break;
                }
#endif
#ifdef _USE_FP16
                case DT_F16: {
                    ret = eltwise_fp16(currentInput, num, loopInner, currentOutput, eltwiseMode);
                    break;
                }
#endif
                default:
                    ret = NOT_SUPPORTED;
                    break;
            }
        }
    }
    return ret;
}
