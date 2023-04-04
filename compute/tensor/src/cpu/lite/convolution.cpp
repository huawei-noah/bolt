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
    ConvolutionParamSpec p,
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
    CHECK_REQUIREMENT(it == 1 && ot == 1);
    U32 group = p.group;
    U32 strideT = p.stride_t;
    U32 strideH = p.stride_h;
    U32 strideW = p.stride_w;
    U32 paddingB = p.pad_before;
    U32 paddingT = p.pad_top;
    U32 paddingL = p.pad_left;
    U32 dilateT = p.dilatedRate_t;
    U32 dilateH = p.dilatedRate_h;
    U32 dilateW = p.dilatedRate_w;
    U32 ocGroupSize = oc / group;
    CHECK_REQUIREMENT(fdf == DF_NCHW);
    for (U32 n = 0; n < in; n++) {
        for (U32 o = 0; o < oc; o++) {
            for (U32 h = 0; h < oh; h++) {
                for (U32 w = 0; w < ow; w++) {
                    T3 value = 0;
                    U32 groupId = o / ocGroupSize;
                    U32 icStart = groupId * fc;
                    U32 icEnd = (groupId + 1) * fc;
                    U32 o_off = ((n * oc + o) * oh + h) * ow + w;
                    for (U32 c = icStart; c < icEnd; c++) {
                        for (I32 fh_idx = 0; fh_idx < (I32)fh; fh_idx++) {
                            I32 ih_idx = h * strideH - paddingT + fh_idx * dilateH;
                            if (ih_idx < 0 || ih_idx >= (I32)ih) {
                                continue;
                            }
                            for (I32 fw_idx = 0; fw_idx < (I32)fw; fw_idx++) {
                                I32 iw_idx = w * strideW - paddingL + fw_idx * dilateW;
                                if (iw_idx < 0 || iw_idx >= (I32)iw) {
                                    continue;
                                }
                                U32 i_off = ((n * ic + c) * ih + ih_idx) * iw + iw_idx;
                                U32 f_off = ((o * fc + c) * fh + fh_idx) * fw + fw_idx;
                                value += inArray[i_off] * filterArray[f_off];
                            }
                        }
                    }
                    outArray[o_off] = value + biasArray[o];
                    CHECK_STATUS(
                        activation_template<T3>(activationDesc, outArray[o_off], &outArray[o_off]));
                }
            }
        }
    }
    return SUCCESS;
}

EE convolution_general(TensorDesc inputDesc,
    void *input,
    void *eltwiseInput,
    TensorDesc filterDesc,
    const void *filter,
    ConvolutionParamSpec p,
    TensorDesc scaleDesc,
    const void *scale,
    TensorDesc biasDesc,
    const void *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    ActivationParamSpec activationDesc)
{
    UNUSED(scaleDesc);
    UNUSED(biasDesc);
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
#ifdef _USE_FP32
        case DT_F32:
            ret = convolution<F32, F32, F32, F32>(inputDesc, (F32 *)input, filterDesc, (F32 *)filter,
                p, (F32 *)bias, (F32 *)scale, outputDesc, (F32 *)output, activationDesc);
            break;
#endif
#ifdef _USE_FP16
        case DT_F16:
            ret = convolution<F16, F16, F16, F16>(inputDesc, (F16 *)input, filterDesc, (F16 *)filter,
                p, (F16 *)bias, (F16 *)scale, outputDesc, (F16 *)output, activationDesc);
            break;
#endif
#ifdef _USE_INT8
        case DT_I8:
            ret = convolution<INT8, F16, F16, F16>(inputDesc, (INT8 *)input, filterDesc,
                (F16 *)filter, p, (F16 *)bias, (F16 *)scale, outputDesc, (F16 *)output,
                activationDesc);
            break;
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
