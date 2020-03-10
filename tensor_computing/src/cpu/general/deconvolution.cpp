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

template<typename T>
inline EE deconvolution(TensorDesc inputDesc, T* inArray,
    TensorDesc filterDesc, const T* filterArray,
    ConvolutionDesc convDesc,
    const T* biasArray,
    TensorDesc outputDesc, T* outArray,
    ActivationMode activationMode)
{
    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));

    if (ic != fc) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 strideH = convDesc.stride_h;
    U32 strideW = convDesc.stride_w;
    U32 paddingT = convDesc.padding_top;
    U32 paddingL = convDesc.padding_left;

    if (idf == DF_NCHWC8) {
        CHECK_STATUS(from_nchwc8_to_nchw<T>(&inputDesc, inArray));
    }
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    if (idf != DF_NCHW) {
        CHECK_STATUS(NOT_MATCH);
    }

    // initialize outputs to 0
    memset(outArray, 0, tensorNumBytes(outputDesc));

    for (U32 n = 0; n < in; n++) {
        for (U32 o = 0; o < oc; o++) {
            for (U32 c = 0; c < ic; c++) {
                for (U32 h = 0; h < ih; h++) {
                    for (U32 w = 0; w < iw; w++) {
                        U32 i_off = n*ic*ih*iw + c*ih*iw + h*iw + w;
                        for (I32 fh_idx = 0; fh_idx < (I32)fh; fh_idx++) {
                            for (I32 fw_idx = 0; fw_idx < (I32)fw; fw_idx++) {
                                I32 oh_idx = fh_idx + strideH * h - paddingT; 
                                I32 ow_idx = fw_idx + strideW * w - paddingL;
                                if (oh_idx >= 0 && oh_idx < (I32)oh && ow_idx >= 0 && ow_idx < (I32)ow) {
                                    U32 o_off = n*oc*oh*ow + o*oh*ow + oh_idx*ow + ow_idx;
                                    U32 f_off = o*ic*fh*fw + c*fh*fw + fh_idx*fw + fw_idx;
                                    outArray[o_off] += inArray[i_off] * filterArray[f_off];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // bias
    for (U32 n = 0; n < in; n++) {
        for (U32 o = 0; o < oc; o++) {
            for (U32 h = 0; h < oh; h++) {
                for (U32 w = 0; w < ow; w++) {
                    U32 o_off = n*oc*oh*ow + o*oh*ow + h*ow + w;
                    U32 b_off = o;
                    outArray[o_off] += biasArray[b_off];
                    switch (activationMode) {
                        case ACTIVATION_NULL: {
                            break;
                        }
                        case ACTIVATION_RELU: {
                            if(outArray[o_off] < 0) {
                                outArray[o_off] = 0;
                            }
                            break;
                        }
                        default:
                            return NOT_SUPPORTED;
                    }
                }
            }
        }
    }

    if (odf == DF_NCHWC8) {
        outputDesc.df = DF_NCHW;
        CHECK_STATUS(from_nchw_to_nchwc8<T>(&outputDesc, outArray));
    }
    return SUCCESS;
}

EE deconvolution_general(TensorDesc inputDesc, void* input,
        TensorDesc filterDesc, const void* filter,
        ConvolutionDesc convDesc,
        TensorDesc scaleDesc, const void* scale,
        TensorDesc biasDesc, const void* bias,
        TensorDesc outputDesc, void* output,
        ActivationMode activationMode)
{
    UNUSED(scaleDesc);
    UNUSED(scale);
    UNUSED(biasDesc);
    
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
#ifdef _USE_FP16
        case DT_F16:
            ret = deconvolution<F16>(inputDesc, (F16*)input,
                                             filterDesc, (F16*)filter,
                                             convDesc,
                                             (F16*)bias,
                                             outputDesc, (F16*)output,
                                             activationMode);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            ret = deconvolution<F32>(inputDesc, (F32*)input,
                                             filterDesc, (F32*)filter,
                                             convDesc,
                                             (F32*)bias,
                                             outputDesc, (F32*)output,
                                             activationMode);
            break;
#endif
        default:
            return NOT_SUPPORTED;
    }
    return ret;
}
