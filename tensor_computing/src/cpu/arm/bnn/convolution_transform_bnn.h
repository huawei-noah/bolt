// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_CONVOLUTION_TRANSFORM_BNN
#define _H_CONVOLUTION_TRANSFORM_BNN

#ifdef _USE_FP16
#include <bitset>
#include <string.h>

#include "type.h"
#include "tensor_desc.h"
#include "error.h"
#include "tensor_computing.h"


inline void bitwise_copy(BIN8 srcVal, U32 srcBit, BIN8* dest, U32 destBit) {
    std::bitset<8> Src(srcVal);
    if (Src.test(srcBit)) {
        *dest |= (1<<destBit); // Set bit
    } else {
        *dest &= ~(1<<destBit); // Clear bit
    }
}

inline EE convolution_transform_filter_bnn(TensorDesc filterDesc, const BIN8* filterArray, TensorDesc *ftmDesc,
    BIN8* ftmArray)
{
    /*
     *  NCHW => (N/16)*(C/8)*(H*W)*n16*c8
     */
    if (nullptr == filterArray || nullptr == ftmDesc || nullptr == ftmArray)
        CHECK_STATUS(NULL_POINTER);

    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    switch (fdf) {
        case DF_NCHWN16C8:
            // Everything is ready
            memcpy(ftmArray, filterArray, fn*fc*fh*fw/8*bytesOf(fdt));
            break;
        case DF_NCHW: {
            /*
             *  NCHW => NCHWN16C8
             *  Now assume fn is divisible by 32
             */
            U32 oc = fn / 16;
            U32 ic = fc / 8;
            for (U32 o = 0; o < oc; o++) {
                for (U32 c = 0; c < ic; c++) {
                    for (U32 hw = 0; hw < fh*fw; hw++) {
                        for (U32 o16 = 0; o16 < 16; o16++) {
                            for (U32 c8 = 0; c8 < 8; c8++) {
                                U32 ftmBitPos = o*fh*fw*ic*128 + c*fh*fw*128 + hw*128 + o16*8 + c8;
                                U32 ftmSlot = ftmBitPos / 8;
                                U32 ftmBitNo = 7 - (ftmBitPos % 8);

                                U32 filterBitPos = (o*16+o16)*ic*8*fh*fw + (c*8+c8)*fh*fw + hw;
                                U32 filterSlot = filterBitPos / 8;
                                U32 filterBitNo = 7 - (filterBitPos % 8);
                                bitwise_copy(filterArray[filterSlot], filterBitNo, ftmArray+ftmSlot, ftmBitNo);
                            }
                        }
                    }
                }
            }
            break;
        }
        default:
            return NOT_MATCH;
    }
    *ftmDesc = tensor4df(fdt, DF_NCHWN16C8, fn, fc, fh, fw);
    return SUCCESS;
}
#endif
#endif
