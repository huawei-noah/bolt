// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifdef _USE_INT8
#include "tensor_computing_type.h"
#include "cpu/arm/int8/tensor_computing_int8.h"

#include "cpu/arm/int8/convolution_winograd.h"
#include "cpu/arm/int8/convolution_gemm.h"

EE convolution_infer_forward_tmp_bytes_int8(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionForwardAlgorithm algorithm, U32 *bytes)
{
    if (nullptr == bytes)
        CHECK_STATUS(NULL_POINTER);
    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 paddingT = convDesc.padding_top;
    U32 paddingB = convDesc.padding_bottom;
    U32 paddingL = convDesc.padding_left;
    U32 paddingR = convDesc.padding_right;

    U32 ih_pad = ih + paddingT + paddingB;
    U32 iw_pad = iw + paddingL + paddingR;

    EE ret = SUCCESS;
    switch (algorithm) {
        case CONVOLUTION_ALGORITHM_WINOGRAD: {
            U32 tile_h = (oh + 3) / 4;
            U32 tile_w = (ow + 3) / 4;
            U32 pad_left = paddingL;
            U32 pad_right = paddingR + (tile_w*4 - ow);
            U32 pad_top = paddingT;
            U32 pad_bottom = paddingB + (tile_h*4 - oh);
            ih_pad = ih + pad_top + pad_bottom;
            iw_pad = iw + pad_left + pad_right;

            *bytes = ic*ih_pad*iw_pad * bytesOf(idt);  // pad
            *bytes += (ic+8)*6*6*12 * bytesOf(DT_F16);  // itm (int16 for int8 inputs) and otm (otm just contains o8 each time)
            *bytes += ic*6*6*12;  // quantized transformed input
            if (odt == DT_I8) {
                *bytes += on*oc*oh*ow*bytesOf(DT_F16);  // Output before quantization
            }
            break;
        }
        case CONVOLUTION_ALGORITHM_GEMM: {
            *bytes = ic*ih_pad*iw_pad + 12*fh*fw*ic + ic*ih*iw;
            if (odt == DT_I8) {
                *bytes += (oc + on*oc*oh*ow)*bytesOf(DT_I32);  // scaled bias + results before quantization
            }
            break;
        }
        default: {
            ret = NOT_MATCH;
            break;
        }
    }
    *bytes += 32;
    return ret;
}

EE convolution_int8(TensorDesc inputDesc, const INT8* input,
    TensorDesc filterDesc, const INT8* filter, F16* scales,
    ConvolutionDesc convDesc, ConvolutionForwardAlgorithm algorithm,
    TensorDesc biasDesc, const F16* bias,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, void* output,
    ActivationMode activationMode,
    Arch arch)
{
    if (nullptr == input || nullptr == filter || nullptr == output || nullptr == bias || nullptr == tmp) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if (idt != DT_I8 && idt != DT_F16)
        CHECK_STATUS(NOT_MATCH);
    if ( fdt != DT_I8 )
        CHECK_STATUS(NOT_MATCH);
    if (odt != DT_F16 && odt != DT_I8)
        CHECK_STATUS(NOT_MATCH);
    if (!((idf == DF_NCHWC8 || (idf == DF_NCHW && (ic == 3 || ic == 1))) && odf == DF_NCHWC8))
        CHECK_STATUS(NOT_MATCH);
    if (!(ic == fc && oc == fn))
        CHECK_STATUS(NOT_MATCH);

    EE ret = SUCCESS;
    switch (algorithm) {
        case CONVOLUTION_ALGORITHM_WINOGRAD:
            ret = convolution_winograd(inputDesc, input, scales, filterDesc, filter, scales+2, convDesc,
                                       biasDesc, bias, tmpBytes, tmp, outputDesc, output, scales+1, activationMode, arch);
            break;
        case CONVOLUTION_ALGORITHM_GEMM:
            ret = convolution_gemm(inputDesc, input, scales, filterDesc, filter, scales+2, convDesc,
                                   biasDesc, bias, tmpBytes, tmp, outputDesc, output, scales+1, activationMode, arch);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
#endif
