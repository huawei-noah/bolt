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

#include "cpu/arm/tensor_computing_arm.h"

EE reshape_arm(TensorDesc inputDesc, void* input,
    TensorDesc outputDesc, void* output)
{
    if (nullptr == input || nullptr == output)
        CHECK_STATUS(NULL_POINTER);

    if (tensorNumElements(inputDesc) != tensorNumElements(outputDesc)) {
        // Only allow the removal of padded convolution channels
        CHECK_REQUIREMENT(DF_NCHWC8 == inputDesc.df);
        CHECK_REQUIREMENT(4 == inputDesc.nDims);
        CHECK_REQUIREMENT(1 == inputDesc.dims[1] && 1 == inputDesc.dims[0]);
        inputDesc.df = DF_NCHW;
    }
    if (DF_NCHWC8 != inputDesc.df) {
        memcpy(output, input, tensorNumBytes(outputDesc));
    } else {
        DataType idt;
        DataFormat idf;
        U32 in, ic, ih, iw;
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));

        U32 elementBytes = bytesOf(idt);
        ic /= 8;
        U8 *inPtr = (U8*)input;
        U8 *outPtr = (U8*)output;
        for (U32 n = 0; n < in; n++) {
            for (U32 c = 0; c < ic; c++) {
                for (U32 hw = 0; hw < ih*iw; hw++) {
                    for (U32 c8 = 0; c8 < 8; c8++) {
                        memcpy(outPtr + elementBytes * (n*ic*8*ih*iw + (c*8 + c8)*ih*iw + hw), 
                                inPtr + elementBytes * (n*ic*ih*iw*8 + c*ih*iw*8 + hw*8 + c8),
                                elementBytes);
                    }
                }
            }
        }
    }
    return SUCCESS;
}
