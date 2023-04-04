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
#include "cpu/tensor_computing_cpu.h"

template <typename T1, typename T2, typename T3, typename T4, int scaleLength>
inline EE deconvolution(TensorDesc inputDesc,
    T1 *inArray,
    TensorDesc filterDesc,
    const T2 *filterArray,
    ConvolutionParamSpec convParamSpec,
    const T4 *biasArray,
    const T3 *scaleArray,
    TensorDesc outputDesc,
    T4 *outArray,
    ActivationParamSpec activationDesc)
{
    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 group = convParamSpec.group;
    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingT = convParamSpec.pad_top;
    U32 paddingL = convParamSpec.pad_left;
    U32 ocGroupSize = oc / group;
    U32 icx = 8;
    U32 ocx = 8;
    if (idf == DF_NCHWC16) {
        icx = 16;
    }
    if (odf == DF_NCHWC16) {
        ocx = 16;
    }

    // initialize outputs to 0
    UNI_MEMSET(outArray, 0, tensorNumElements(outputDesc) * sizeof(T4));
    U32 ic8 = ic / icx;
    U32 oc8 = oc / ocx;
    for (U32 n = 0; n < in; n++) {
        for (U32 o = 0; o < oc; o++) {
            U32 groupId = o / ocGroupSize;
            U32 icStart = groupId * fn;
            U32 icEnd = (groupId + 1) * fn;
            for (U32 c = icStart; c < icEnd; c++) {
                for (U32 h = 0; h < ih; h++) {
                    for (U32 w = 0; w < iw; w++) {
                        U32 i_off;
                        if ((idf != DF_NCHWC8) && (idf != DF_NCHWC16)) {
                            i_off = ((n * ic + c) * ih + h) * iw + w;
                        } else {
                            i_off = (((n * ic8 + (c / icx)) * ih + h) * iw + w) * icx + c % icx;
                        }
                        for (I32 fh_idx = 0; fh_idx < (I32)fh; fh_idx++) {
                            for (I32 fw_idx = 0; fw_idx < (I32)fw; fw_idx++) {
                                I32 oh_idx = fh_idx + strideH * h - paddingT;
                                I32 ow_idx = fw_idx + strideW * w - paddingL;
                                if (oh_idx >= 0 && oh_idx < (I32)oh && ow_idx >= 0 &&
                                    ow_idx < (I32)ow) {
                                    U32 o_off;
                                    if ((odf != DF_NCHWC8) && (odf != DF_NCHWC16)) {
                                        o_off = ((n * oc + o) * oh + oh_idx) * ow + ow_idx;
                                    } else {
                                        o_off =
                                            (((n * oc8 + (o / ocx)) * oh + oh_idx) * ow + ow_idx) * ocx +
                                            o % ocx;
                                    }
                                    U32 f_off =
                                        (((c - icStart) * fc + o) * fh + fh_idx) * fw + fw_idx;
                                    outArray[o_off] += inArray[i_off] * (T4)filterArray[f_off];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    T4 scale = 1;
    if (scaleLength == 1) {
        scale = scaleArray[0];
    }
    // bias
    U32 ohow = oh * ow;
    for (U32 i = 0; i < tensorNumElements(outputDesc); i++) {
        U32 o;
        if ((odf != DF_NCHWC8) && (odf != DF_NCHWC16)) {
            o = (i / ohow) % oc;
        } else {
            o = (i / (ohow * ocx)) % oc8 * ocx + i % ocx;
        }
        outArray[i] = scale * outArray[i] + biasArray[o];
        switch (activationDesc.mode) {
            case ACTIVATION_NULL: {
                break;
            }
            case ACTIVATION_RELU: {
                F32 tmp = activationDesc.value[0] * outArray[i];
                if (outArray[i] < tmp) {
                    outArray[i] = tmp;
                }
                break;
            }
            default:
                return NOT_SUPPORTED;
        }
    }
    return SUCCESS;
}

EE deconvolution_general(TensorDesc inputDesc,
    void *input,
    TensorDesc filterDesc,
    const void *filter,
    ConvolutionParamSpec convParamSpec,
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
    UNUSED(scale);
    UNUSED(biasDesc);

    EE ret = SUCCESS;
    switch (inputDesc.dt) {
#ifdef _USE_INT8
        case DT_I8:{
            F32 scaleI = ((F32 *)scale)[0];
            if (inputDesc.dt != DT_I8) {
                TensorDesc qDesc = inputDesc;
                qDesc.dt = DT_I8;
                CHECK_STATUS(quantize_cpu(inputDesc, input, &qDesc, tmp, &scaleI, CPU_GENERAL));
                input = (void *)tmp;
                inputDesc = qDesc;
                tmp = (void *)((INT8 *)tmp + tensorNumBytes(inputDesc));
            }
            F32 scaleN = 1.0;
            TensorDesc tmpDesc = outputDesc;
            void *tmpOutput;
            if (outputDesc.dt == DT_I8) {
#ifdef _USE_FP16
                tmpDesc.dt = DT_F16;
#else
                tmpDesc.dt = DT_F32;
#endif
                tmpOutput = tmp;
            } else {
                tmpOutput = output;
            }
#ifdef _USE_FP16
            ret = deconvolution<INT8, INT8, F32, F16, 1>(inputDesc, (INT8 *)input, filterDesc, (INT8 *)filter,
                convParamSpec, (F16 *)bias, &scaleN, outputDesc, (F16 *)output, activationDesc);
#else
            ret = deconvolution<INT8, INT8, F32, F32, 1>(inputDesc, (INT8 *)input, filterDesc, (INT8 *)filter,
                convParamSpec, (F32 *)bias, &scaleN, outputDesc, (F32 *)output, activationDesc);
#endif
            if (outputDesc.dt == DT_I8) {
                CHECK_STATUS(quantize_cpu(
                    tmpDesc, tmpOutput, &outputDesc, output, (F32 *)scale + 1, CPU_GENERAL));
            }
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16:
            ret = deconvolution<F16, F16, F16, F16, 0>(inputDesc, (F16 *)input, filterDesc, (F16 *)filter,
                convParamSpec, (F16 *)bias, nullptr, outputDesc, (F16 *)output, activationDesc);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            ret = deconvolution<F32, F32, F32, F32, 0>(inputDesc, (F32 *)input, filterDesc, (F32 *)filter,
                convParamSpec, (F32 *)bias, nullptr, outputDesc, (F32 *)output, activationDesc);
            break;
#endif
        default:
            return NOT_SUPPORTED;
    }
    return ret;
}
