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
#include "cpu/general/general_functions.h"

template <typename T1, typename T2, typename T3, typename T4>
inline EE convolution(TensorDesc inputDesc,
    T1 *inArray,
    TensorDesc filterDesc,
    const T2 *filterArray,
    ConvolutionParamSpec convParamSpec,
    const T3 *biasArray,
    const T4 *scaleArray,
    TensorDesc outputDesc,
    T4 *outArray,
    ActivationParamSpec activationDesc,
    T1 paddingValue = 0)
{
    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, it, ih, iw;
    U32 fn, fc, ft, fh, fw;
    U32 on, oc, ot, oh, ow;
    it = ft = ot = 1;
    if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    } else if (tensorIs5d(inputDesc)) {
        CHECK_STATUS(tensor5dGet(inputDesc, &idt, &idf, &in, &ic, &it, &ih, &iw));
        CHECK_STATUS(tensor5dGet(filterDesc, &fdt, &fdf, &fn, &fc, &ft, &fh, &fw));
        CHECK_STATUS(tensor5dGet(outputDesc, &odt, &odf, &on, &oc, &ot, &oh, &ow));
    } else {
        return NOT_SUPPORTED;
    }
    U32 group = convParamSpec.group;
    U32 strideT = convParamSpec.stride_t;
    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingB = convParamSpec.padding_before;
    U32 paddingT = convParamSpec.padding_top;
    U32 paddingL = convParamSpec.padding_left;
    U32 dilateT = convParamSpec.dilatedRate_t;
    U32 dilateH = convParamSpec.dilatedRate_h;
    U32 dilateW = convParamSpec.dilatedRate_w;
    U32 ocGroupSize = oc / group;
    CHECK_REQUIREMENT(fdf == DF_NCHW);

    // For BNN, accumulated values are always 0 or 1, which may lead to error if buf is floating point.
    U32 ic8 = ic / 8;
    U32 oc8 = oc / 8;
    for (U32 n = 0; n < in; n++) {
        for (U32 o = 0; o < oc; o++) {
            for (U32 t = 0; t < ot; t++) {
                for (U32 h = 0; h < oh; h++) {
                    for (U32 w = 0; w < ow; w++) {
                        T3 value = 0;
                        U32 groupId = o / ocGroupSize;
                        U32 icStart = groupId * fc;
                        U32 icEnd = (groupId + 1) * fc;
                        for (U32 c = icStart, f_off = o * fc * ft * fh * fw; c < icEnd; c++) {
                            for (I32 ft_idx = 0; ft_idx < (I32)ft; ft_idx++) {
                                for (I32 fh_idx = 0; fh_idx < (I32)fh; fh_idx++) {
                                    for (I32 fw_idx = 0; fw_idx < (I32)fw; fw_idx++, f_off++) {
                                        I32 it_idx = t * strideT - paddingB + ft_idx * dilateT;
                                        I32 ih_idx = h * strideH - paddingT + fh_idx * dilateH;
                                        I32 iw_idx = w * strideW - paddingL + fw_idx * dilateW;
                                        if (it_idx >= 0 && it_idx < (I32)it && ih_idx >= 0 &&
                                            ih_idx < (I32)ih && iw_idx >= 0 && iw_idx < (I32)iw) {
                                            U32 i_off;
                                            if (idf == DF_NCHW) {
                                                i_off = (((n * ic + c) * it + it_idx) * ih + ih_idx) *
                                                        iw +
                                                    iw_idx;
                                            } else {
                                                i_off = ((((n * ic8 + (c / 8)) * it + it_idx) * ih +
                                                             ih_idx) *
                                                                iw +
                                                            iw_idx) *
                                                        8 +
                                                    c % 8;
                                            }
                                            value += inArray[i_off] * filterArray[f_off];
                                        } else {
                                            value += paddingValue * filterArray[f_off];
                                        }
                                    }
                                }
                            }
                        }
                        U32 o_off;
                        if (odf == DF_NCHW) {
                            o_off = (((n * oc + o) * ot + t) * oh + h) * ow + w;
                        } else {
                            o_off = ((((n * oc8 + (o / 8)) * ot + t) * oh + h) * ow + w) * 8 + o % 8;
                        }

                        T4 scale = 1;
                        if (scaleArray != nullptr) {
                            scale = scaleArray[o];
                        }
                        outArray[o_off] += scale * value + biasArray[o];
                        CHECK_STATUS(activation_template<T3>(
                            activationDesc, outArray[o_off], &outArray[o_off]));
                    }
                }
            }
        }
    }
    return SUCCESS;
}

#ifdef _USE_FP16
void bnn_input_process(TensorDesc inputDesc, F16 *input, DataType fdt, short *output)
{
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
        if (input[i] >= centerValue) {
            output[i] = 1;
        } else {
            output[i] = zeroValue;
        }
    }
}

void bnn_filter_process(TensorDesc filterDesc, BIN8 *filter, short *filterTransformed)
{
    short zeroValue = 0;
    if (filterDesc.dt == DT_BIN11) {
        zeroValue = -1;
    }
    U32 len = tensorNumElements(filterDesc);
    for (U32 i = 0; i < len; i++) {
        U32 bitSlot = i / 8;
        U32 bitNo = 7 - (i % 8);
        std::bitset<8> Q(filter[bitSlot]);
        if (Q.test(bitNo)) {
            filterTransformed[i] = 1;
        } else {
            filterTransformed[i] = zeroValue;
        }
    }
}
#endif

EE convolution_general(TensorDesc inputDesc,
    void *input,
    void *eltwiseInput,
    TensorDesc filterDesc,
    const void *filter,
    ConvolutionParamSpec convParamSpec,
    TensorDesc scaleDesc,
    const void *scale,
    TensorDesc biasDesc,
    const void *bias,
    TensorDesc outputDesc,
    void *output,
    ActivationParamSpec activationDesc)
{
    UNUSED(scaleDesc);
    UNUSED(biasDesc);

    if (eltwiseInput == nullptr) {
        memset(output, 0, tensorNumBytes(outputDesc));
    } else {
        memcpy(output, eltwiseInput, tensorNumBytes(outputDesc));
    }

    EE ret = SUCCESS;
    switch (filterDesc.dt) {
#ifdef _USE_FP32
        case DT_F32:
            ret = convolution<F32, F32, F32, F32>(inputDesc, (F32 *)input, filterDesc, (F32 *)filter,
                convParamSpec, (F32 *)bias, (F32 *)scale, outputDesc, (F32 *)output, activationDesc);
            break;
#endif
#ifdef _USE_FP16
        case DT_F16:
            ret = convolution<F16, F16, F16, F16>(inputDesc, (F16 *)input, filterDesc, (F16 *)filter,
                convParamSpec, (F16 *)bias, (F16 *)scale, outputDesc, (F16 *)output, activationDesc);
            break;
#endif
#ifdef _USE_INT8
        case DT_I8:
            ret = convolution<INT8, F16, F16, F16>(inputDesc, (INT8 *)input, filterDesc,
                (F16 *)filter, convParamSpec, (F16 *)bias, (F16 *)scale, outputDesc, (F16 *)output,
                activationDesc);
            break;
#endif
#ifdef _USE_FP16
        case DT_BIN01: {
            std::vector<short> inputTransformed(tensorNumElements(inputDesc));
            std::vector<short> filterTransformed(tensorNumElements(filterDesc));
            bnn_input_process(inputDesc, (F16 *)input, filterDesc.dt, inputTransformed.data());
            bnn_filter_process(filterDesc, (BIN8 *)filter, filterTransformed.data());
            ret = convolution<short, short, F16, F16>(inputDesc, inputTransformed.data(),
                filterDesc, filterTransformed.data(), convParamSpec, (F16 *)bias, (F16 *)scale,
                outputDesc, (F16 *)output, activationDesc, 0);
            break;
        }
        case DT_BIN11: {
            std::vector<short> inputTransformed(tensorNumElements(inputDesc));
            std::vector<short> filterTransformed(tensorNumElements(filterDesc));
            bnn_input_process(inputDesc, (F16 *)input, filterDesc.dt, inputTransformed.data());
            bnn_filter_process(filterDesc, (BIN8 *)filter, filterTransformed.data());
            ret = convolution<short, short, F16, F16>(inputDesc, inputTransformed.data(),
                filterDesc, filterTransformed.data(), convParamSpec, (F16 *)bias, (F16 *)scale,
                outputDesc, (F16 *)output, activationDesc, -1);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
