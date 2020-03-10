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

template<typename T1, typename T2, typename T3>
inline EE depthwise_convolution(TensorDesc inputDesc, T1* inArray,
    TensorDesc filterDesc, const T2* filterArray,
    ConvolutionDesc convDesc,
    const T3* biasArray,
    TensorDesc outputDesc, T3* outArray,
    ActivationMode depthwiseActivationMode,
    ActivationMode pointwiseActivationMode)
{
    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 strideH = convDesc.stride_h;
    U32 strideW = convDesc.stride_w;
    U32 paddingT = convDesc.padding_top;
    U32 paddingL = convDesc.padding_left;
    bool fuseDepthwisePointwise = (fdf == DF_CHW_NC) ? true : false;

    if (idf == DF_NCHWC8)
        CHECK_STATUS(from_nchwc8_to_nchw<T1>(&inputDesc, inArray));
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    if (idf != DF_NCHW)
        CHECK_STATUS(NOT_MATCH);

    T3* pwArray;
    if (fuseDepthwisePointwise) {
        pwArray = (T3*)malloc(ic * oh * ow * sizeof(T3));
        memset(pwArray, 0, ic * oh * ow * sizeof(T3));
    }
    else {
        pwArray = outArray;
    }
    const T1* filterDwArray = (const T1*)filterArray;
    const T1* filterPwArray = (const T1*)filterArray + fh*fw*ic;
    for (U32 n = 0; n < in; n++) {
        // dw conv
        for (U32 c = 0; c < ic; c++) {
            for (U32 h = 0; h < oh; h++) {
                for (U32 w = 0; w < ow; w++) {
                    for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                        for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                            I32 ih_idx = h * strideH - paddingT + fh_idx;
                            I32 iw_idx = w * strideW - paddingL + fw_idx;
                            if (ih_idx >= 0 && ih_idx < (I32)ih && iw_idx >= 0 && iw_idx < (I32)iw) {
                                pwArray[c*oh*ow + h*ow + w] +=
                                    inArray[n*ic*ih*iw + c*ih*iw + ih_idx*iw + iw_idx] *
                                    filterDwArray[c*fh*fw + fh_idx*fw +fw_idx];
                            }
                        }
                    }
                }
            }
        }
        // bias
        for (U32 c = 0; c < ic; c++) {
            for (U32 hw = 0; hw < oh*ow; hw++) {
                U32 pw_off = c*oh*ow + hw;
                U32 b_off = c;
                pwArray[pw_off] += biasArray[b_off];
                CHECK_STATUS(activation<T3>(depthwiseActivationMode, pwArray[pw_off], &pwArray[pw_off]));
            }
        }
        if (fuseDepthwisePointwise) {
            // pw conv
            for (U32 o = 0; o < oc; o++) {
                for (U32 c = 0; c < ic; c++) {
                    for (U32 hw = 0; hw < oh*ow; hw++) {
                        outArray[n*oc*oh*ow + o*oh*ow + hw] += pwArray[c*oh*ow + hw] *
                            filterPwArray[o*ic + c];
                    }
                }
            }
            // bias
            for (U32 o = 0; o < oc; o++) {
                for (U32 h = 0; h < oh; h++) {
                    for (U32 w = 0; w < ow; w++) {
                        U32 o_off = n*oc*oh*ow + o*oh*ow + h*ow + w;
                        U32 b_off = ic + o;
                        outArray[o_off] += biasArray[b_off];
                        CHECK_STATUS(activation<T3>(pointwiseActivationMode, outArray[o_off], &outArray[o_off]));
                    }
                }
            }
        }
    }

    if(fuseDepthwisePointwise)
        free(pwArray);

    if (odf == DF_NCHWC8) {
        outputDesc.df = DF_NCHW;
        CHECK_STATUS(from_nchw_to_nchwc8<T3>(&outputDesc, outArray));
    }
    return SUCCESS;
}

EE depthwise_convolution_general(TensorDesc inputDesc, void* input,
        TensorDesc filterDesc, const void* filter,
        ConvolutionDesc convDesc,
        TensorDesc biasDesc, const void* bias,
        TensorDesc outputDesc, void* output,
        ActivationMode depthwiseActivationMode,
        ActivationMode pointwiseActivationMode)
{
    UNUSED(biasDesc);

    EE ret = SUCCESS;
    switch (inputDesc.dt) {
#ifdef _USE_FP16
        case DT_F16:
            ret = depthwise_convolution<F16, F16, F16>(inputDesc, (F16*)input,
                                                       filterDesc, (F16*)filter,
                                                       convDesc,
                                                       (F16*)bias,
                                                       outputDesc, (F16*)output,
                                                       depthwiseActivationMode,
                                                       pointwiseActivationMode);
            break;
#endif
#ifdef _USE_INT8
        case DT_I8:
            ret = depthwise_convolution<INT8, I32, I32>(inputDesc, (INT8*)input,
                                                        filterDesc, (I32*)filter,
                                                        convDesc,
                                                        (I32*)bias,
                                                        outputDesc, (I32*)output,
                                                        depthwiseActivationMode,
                                                        pointwiseActivationMode);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            ret = depthwise_convolution<F32, F32, F32>(inputDesc, (F32*)input,
                                                        filterDesc, (F32*)filter,
                                                        convDesc,
                                                        (F32*)bias,
                                                        outputDesc, (F32*)output,
                                                        depthwiseActivationMode,
                                                        pointwiseActivationMode);
            break;
#endif
        default:
            return NOT_SUPPORTED;
    }
    return ret;
}
