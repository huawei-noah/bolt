// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <math.h>
#include <bitset>
#include "type.h"
#include "tensor_desc.h"
#include "error.h"
#include "tensor_computing_type.h"
#include "cpu/general/tensor_computing_general.h"
#include "cpu/general/common_general.h"

template<typename T1, typename T2, typename T3, typename T4>
inline EE convolution(TensorDesc inputDesc, T1* inArray,
    TensorDesc filterDesc, const T2* filterArray,
    ConvolutionDesc convDesc,
    const T3* biasArray,
    const T4* scaleArray,
    TensorDesc outputDesc, T4* outArray,
    ActivationMode activationMode,
    T1 paddingValue=0)
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
    U32 dilateH = convDesc.dilatedRate_h;
    U32 dilateW = convDesc.dilatedRate_w;

    if (idf == DF_NCHWC8)
        CHECK_STATUS(from_nchwc8_to_nchw<T1>(&inputDesc, inArray));
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    if (idf != DF_NCHW)
        CHECK_STATUS(NOT_MATCH);

    // For BNN, accumulated values are always 0 or 1, which may lead to error if buf is floating point.
    std::vector<T1> outBuf(tensorNumElements(outputDesc));

    for (U32 n = 0; n < in; n++) {
        for (U32 o = 0; o < oc; o++) {
             for (U32 h = 0; h < oh; h++) {
                for (U32 w = 0; w < ow; w++) {
                    U32 o_off = n*oc*oh*ow + o*oh*ow + h*ow + w;
                    outBuf[o_off] = 0;
                    for (U32 c = 0; c < ic; c++) {
                        for (I32 fh_idx = 0; fh_idx < (I32)fh; fh_idx++) {
                            for (I32 fw_idx = 0; fw_idx < (I32)fw; fw_idx++) {
                                I32 ih_idx = h * strideH - paddingT + fh_idx*dilateH;
                                I32 iw_idx = w * strideW - paddingL + fw_idx*dilateW;
                                U32 f_off = o*ic*fh*fw + c*fh*fw + fh_idx*fw + fw_idx;
                                if (ih_idx >= 0 && ih_idx < (I32)ih && iw_idx >= 0 && iw_idx < (I32)iw) {
                                    U32 i_off = n*ic*ih*iw + c*ih*iw + ih_idx*iw + iw_idx;
                                    outBuf[o_off] += inArray[i_off] * filterArray[f_off];
                                }
                                else {
                                    outBuf[o_off] += paddingValue * filterArray[f_off];
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
                    T4 scale = 1;
                    if (scaleArray != nullptr)
                        scale = scaleArray[b_off];
                    outArray[o_off] = scale * outBuf[o_off] + biasArray[b_off];
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
                        case ACTIVATION_SIGMOID: {
                            outArray[o_off] = 1.0f / (1.0f + exp(-1 * outArray[o_off]));
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
        CHECK_STATUS(from_nchw_to_nchwc8<T3>(&outputDesc, outArray));
    }
    return SUCCESS;
}

#ifdef _USE_FP16
void bnn_input_process(TensorDesc inputDesc, F16 *input, DataType fdt, short *output) {
    F16 centerValue = 0.0;
    if (fdt == DT_BIN01) {
        centerValue = 0.5;
    }
    short zeroValue = 0;
    if (fdt == DT_BIN11) {
        zeroValue = -1;
    }
    U32 len = tensorNumElements(inputDesc);
    for (U32 i = 0; i < len; i++) {
        if (input[i] >= centerValue)
            output[i] = 1;
        else
            output[i] = zeroValue;
    }
}

void bnn_filter_process(TensorDesc filterDesc, BIN8 *filter, short *filterTransformed) {
    short zeroValue = 0;
    if (filterDesc.dt == DT_BIN11) {
        zeroValue = -1;
    }
    U32 len = tensorNumElements(filterDesc);
    for (U32 i = 0; i < len; i++) {
        U32 bitSlot = i / 8;
        U32 bitNo = 7 - (i%8);
        std::bitset<8> Q(filter[bitSlot]);
        if (Q.test(bitNo)) {
            filterTransformed[i] = 1;
        } else {
            filterTransformed[i] = zeroValue;
        }
    }
}
#endif

EE convolution_general(TensorDesc inputDesc, void* input,
        TensorDesc filterDesc, const void* filter,
        ConvolutionDesc convDesc,
        TensorDesc scaleDesc, const void* scale,
        TensorDesc biasDesc, const void* bias,
        TensorDesc outputDesc, void* output,
        ActivationMode activationMode)
{
    UNUSED(scaleDesc);
    UNUSED(biasDesc);
    
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
#ifdef _USE_FP32
        case DT_F32:
            ret = convolution<F32, F32, F32, F32>(inputDesc, (F32*)input,
                                                  filterDesc, (F32*)filter,
                                                  convDesc,
                                                  (F32*)bias,
                                                  (F32*)scale,
                                                  outputDesc, (F32*)output,
                                                  activationMode);
            break;
#endif
#ifdef _USE_FP16
        case DT_F16:
            ret = convolution<F16, F16, F16, F16>(inputDesc, (F16*)input,
                                                  filterDesc, (F16*)filter,
                                                  convDesc,
                                                  (F16*)bias,
                                                  (F16*)scale,
                                                  outputDesc, (F16*)output,
                                                  activationMode);
            break;
#endif
#ifdef _USE_INT8
        case DT_I8:
            ret = convolution<INT8, F16, F16, F16>(inputDesc, (INT8*)input,
                                                   filterDesc, (F16*)filter,
                                                   convDesc,
                                                   (F16*)bias,
                                                   (F16*)scale,
                                                   outputDesc, (F16*)output,
                                                   activationMode);
            break;
#endif
#ifdef _USE_FP16
        case DT_BIN01: {
            std::vector<short> inputTransformed(tensorNumElements(inputDesc));
            std::vector<short> filterTransformed(tensorNumElements(filterDesc));
            bnn_input_process(inputDesc, (F16*)input, filterDesc.dt, inputTransformed.data());
            bnn_filter_process(filterDesc, (BIN8*)filter, filterTransformed.data());
            ret = convolution<short, short, F16, F16>(inputDesc, inputTransformed.data(),
                                                      filterDesc, filterTransformed.data(),
                                                      convDesc,
                                                      (F16*)bias,
                                                      (F16*)scale,
                                                      outputDesc, (F16*)output,
                                                      activationMode, 0);
            break;
        }
        case DT_BIN11: {
            std::vector<short> inputTransformed(tensorNumElements(inputDesc));
            std::vector<short> filterTransformed(tensorNumElements(filterDesc));
            bnn_input_process(inputDesc, (F16*)input, filterDesc.dt, inputTransformed.data());
            bnn_filter_process(filterDesc, (BIN8*)filter, filterTransformed.data());
            ret = convolution<short, short, F16, F16>(inputDesc, inputTransformed.data(),
                                                      filterDesc, filterTransformed.data(),
                                                      convDesc,
                                                      (F16*)bias,
                                                      (F16*)scale,
                                                      outputDesc, (F16*)output,
                                                      activationMode, -1);
            break;
        }
#endif
        default:
            return NOT_SUPPORTED;
    }
    return ret;
}
