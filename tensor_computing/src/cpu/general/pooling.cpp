// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <float.h>
#include "type.h"
#include "tensor_desc.h"
#include "error.h"
#include "tensor_computing_type.h"
#include "cpu/general/tensor_computing_general.h"
#include "cpu/general/common_general.h"


template<typename T>
EE pooling(T *input, T* output,
           U32 in, U32 ic, U32 ih, U32 iw,
           U32 strideH, U32 strideW, U32 paddingT, U32 paddingB, U32 paddingL, U32 paddingR, U32 kernelH, U32 kernelW,
           PoolingMode pm, RoundMode rm,
           U32 alignSize,
           F32 minValue)
{
    U32 oh = 0, ow = 0;
    if (rm == CEIL) {
        oh = (U32)(ceil((double(ih + paddingT + paddingB - kernelH) / strideH))) + 1;
        ow = (U32)(ceil((double(iw + paddingL + paddingR - kernelW) / strideW))) + 1;
    }
    if (rm == FLOOR) {
        oh = (U32)(floor((double(ih + paddingT + paddingB - kernelH) / strideH))) + 1;
        ow = (U32)(floor((double(iw + paddingL + paddingR - kernelW) / strideW))) + 1;
    }

    CHECK_REQUIREMENT(ic % alignSize == 0);
    ic = ic / alignSize;

    for (U32 n=0; n<in; n++){
        for (U32 c=0; c<ic; c++){
            for (U32 j=0; j<alignSize; j++){
                for (I32 h=0; h<(I32)oh; h++){
                    for (I32 w=0; w<(I32)ow; w++){
                        int hstart = int(h * strideH - paddingT);
                        int wstart = int(w * strideW - paddingL);
                        int hend = hstart + kernelH;
                        int wend = wstart + kernelW;
                        hstart = (hstart < 0) ? 0 : hstart;
                        wstart = (wstart < 0) ? 0 : wstart;
                        hend = (hend > (int)ih) ? ih : hend;
                        wend = (wend > (int)iw) ? iw : wend;
                        float poolSize = (hend - hstart)*(wend - wstart);

                        T value;
                        switch(pm){
                            case POOLING_MAX:
                                value = minValue;
                                break;
                            case POOLING_MEAN:
                                value = 0;
                                break;
                            default:
                                return NOT_SUPPORTED;
                        }
                        for (int x = hstart; x < hend; x++) {
                            for (int y = wstart; y < wend; y++) {
                                U32 in_off = ((((n*ic + c)*ih) + x)*iw + y)*alignSize + j;
                                switch(pm){
                                    case POOLING_MAX:
                                        value = (value > input[in_off]) ? value : input[in_off];
                                        break;
                                    case POOLING_MEAN:
                                        value += input[in_off];
                                        break;
                                    default:
                                        return NOT_SUPPORTED;
                                }
                            }
                        }
                        switch(pm){
                            case POOLING_MAX:
                                break;
                            case POOLING_MEAN:
                                value = value / poolSize;
                                break;
                            default:
                                return NOT_SUPPORTED;
                        }

                        U32 out_off = ((((n*ic + c)*oh) + h)*ow + w)*alignSize + j;
                        output[out_off] = value;
                    }
                }
            }
        }
    }
    return SUCCESS;
}

EE pooling_general(TensorDesc inputDesc, const void* input, PoolingDesc poolingDesc, TensorDesc outputDesc, void* output)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in = 0, ic = 0, ih = 0, iw = 0,
        on = 0, oc = 0, oh = 0, ow = 0;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if (in != on || ic != oc) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (idf != DF_NCHWC8 || odf != idf) {
        CHECK_STATUS(NOT_MATCH);
    }

    U32 strideH = poolingDesc.stride_h;
    U32 strideW = poolingDesc.stride_w;
    U32 paddingT = poolingDesc.padding_top;
    U32 paddingB = poolingDesc.padding_bottom;
    U32 paddingL = poolingDesc.padding_left;
    U32 paddingR = poolingDesc.padding_right;
    U32 kernelSizeH = poolingDesc.kernelSize_h;
    U32 kernelSizeW = poolingDesc.kernelSize_w;

    EE ret = SUCCESS;
    switch (idt) {
#ifdef _USE_FP32
        case DT_F32:
            ret = pooling((F32*)input, (F32*)output,
                          in, ic, ih, iw,
                          strideH, strideW, paddingT, paddingB, paddingL, paddingR,
                          kernelSizeH, kernelSizeW,
                          poolingDesc.pm, poolingDesc.rm,
                          8, FLT_MIN);
            break;
#endif
#ifdef _USE_FP16
        case DT_F16:
            ret = pooling((F16*)input, (F16*)output,
                          in, ic, ih, iw,
                          strideH, strideW, paddingT, paddingB, paddingL, paddingR,
                          kernelSizeH, kernelSizeW,
                          poolingDesc.pm, poolingDesc.rm,
                          8, UNI_F16_MIN);
            break;
#endif
        default:
            ret = NOT_SUPPORTED;
    }
    return ret;
}
