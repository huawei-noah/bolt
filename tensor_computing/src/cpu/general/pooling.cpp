// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "type.h"
#include "tensor_desc.h"
#include "error.h"
#include "tensor_computing_type.h"
#include "cpu/general/tensor_computing_general.h"
#include "cpu/general/common_general.h"

EE pooling(F16 *input, F16* output,
             U32 in, U32 ic, U32 ih, U32 iw,
             U32 stride, U32 padding, U32 kernel_H, U32 kernel_W,
             PoolingMode pm, RoundMode rm,
             U32 alignSize)
{
    F16 F16_MIN = -65504;
    U32 oh = 0, ow = 0;
    if (rm == CEIL) {
        oh = (U32)(ceil((double(ih + 2.0 * padding - kernel_H) / stride))) + 1;
        ow = (U32)(ceil((double(iw + 2.0 * padding - kernel_W) / stride))) + 1;
    }
    if (rm == FLOOR) {
        oh = (U32)(floor((double(ih + 2.0 * padding - kernel_H) / stride))) + 1;
        ow = (U32)(floor((double(iw + 2.0 * padding - kernel_W) / stride))) + 1;
    }

    assert(ic % alignSize == 0);
    ic = ic / alignSize;

    for (U32 n=0; n<in; n++){
        for (U32 c=0; c<ic; c++){
            for (U32 j=0; j<alignSize; j++){
                for (I32 h=0; h<(I32)oh; h++){
                    for (I32 w=0; w<(I32)ow; w++){
                        int hstart = int(h * stride - padding);
                        int wstart = int(w * stride - padding);
                        int hend = hstart + kernel_H;
                        int wend = wstart + kernel_W;
                        hstart = (hstart < 0) ? 0 : hstart;
                        wstart = (wstart < 0) ? 0 : wstart;
                        hend = (hend > (int)ih) ? ih : hend;
                        wend = (wend > (int)iw) ? iw : wend;
                        float poolSize = (hend - hstart)*(wend - wstart);

                        F16 value;
                        switch(pm){
                            case Max:
                                value = F16_MIN;
                                break;
                            case Mean:
                                value = 0;
                                break;
                            default:
                                return NOT_SUPPORTED;
                        }
                        for (int x = hstart; x < hend; x++) {
                            for (int y = wstart; y < wend; y++) {
                                U32 in_off = ((((n*ic + c)*ih) + x)*iw + y)*alignSize + j;
                                switch(pm){
                                    case Max:
                                        value = (value > input[in_off]) ? value : input[in_off];
                                        break;
                                    case Mean:
                                        value += input[in_off];
                                        break;
                                    default:
                                        return NOT_SUPPORTED;
                                }
                            }
                        }
                        switch(pm){
                            case Max:
                                value = value;
                                break;
                            case Mean:
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
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);
    }
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in = 0, ic = 0, ih = 0, iw = 0,
        on = 0, oc = 0, oh = 0, ow = 0;
    CHECK_STATUS_WITH_RETURN(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS_WITH_RETURN(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if (in != on || ic != oc) {
        CHECK_STATUS_WITH_RETURN(NOT_MATCH);
    }
    if (idf != DF_NCHWC8 || odf != idf) {
        CHECK_STATUS_WITH_RETURN(NOT_MATCH);
    }

    U32 stride = poolingDesc.stride;
    U32 padding = poolingDesc.padding;
    U32 kernelSize = poolingDesc.kernelSize;

    EE ret = SUCCESS;
    switch (idt) {
        case DT_F16:
            pooling((F16*)input, (F16*)output,
                             in, ic, ih, iw,
                             stride, padding, kernelSize, kernelSize,
                             poolingDesc.pm, poolingDesc.rm,
                             8);
            break;
        default:
            return NOT_SUPPORTED;
    }
    return ret;
}
