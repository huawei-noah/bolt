// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "cpu/arm/bnn/convolution_bnn.h"
#include "cpu/arm/bnn/convolution_xnor.h"
#include "cpu/arm/bnn/convolution_dorefa.h"


EE convolution_infer_forward_algorithm_bnn(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionPolicy policy, ConvolutionForwardAlgorithm *algorithm)
{
    UNUSED(convDesc);
    UNUSED(policy);

    if (nullptr == algorithm)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);
    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS_WITH_RETURN(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS_WITH_RETURN(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS_WITH_RETURN(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if (DT_DOREFA != fdt && DT_XNOR != fdt) {
        CHECK_STATUS_WITH_RETURN(NOT_MATCH);
    }

    if (DT_F16 != idt && DT_F16 != odt) {
        CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
    }

    if (fn % 32 != 0) {
        CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
    }

    *algorithm = CONVOLUTION_ALGORITHM_BNN;
    return SUCCESS;
}

EE convolution_infer_forward_tmp_bytes_bnn(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionForwardAlgorithm algorithm, U32 *bytes)
{
    UNUSED(outputDesc);
    
    if (nullptr == bytes)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);
    DataType idt, fdt;
    DataFormat idf, fdf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    CHECK_STATUS_WITH_RETURN(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS_WITH_RETURN(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    U32 padding = convDesc.padding;

    U32 ih_pad = ih + 2*padding;
    U32 iw_pad = iw + 2*padding;
    EE ret = SUCCESS;
    switch (algorithm) {
        case CONVOLUTION_ALGORITHM_BNN:
            *bytes = ic*ih_pad*iw_pad + 8*fh*fw*ic + ic*ih*iw;
            break;
        default:
            ret = NOT_MATCH;
            break;
    }
    *bytes /= 8;
    *bytes *= sizeof(BIN8);
    return ret;
}

EE convolution_bnn(TensorDesc inputDesc, const F16* input,
    TensorDesc filterDesc, const BIN8* filter,
    ConvolutionDesc convDesc,
    TensorDesc scaleDesc, const F16* scale,
    TensorDesc biasDesc, const F16* bias,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, F16* output,
    ActivationMode activationMode,
    Arch arch)
{
    if(nullptr == input || nullptr == filter || nullptr == output || nullptr == scale || nullptr == bias || nullptr == tmp)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);
    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS_WITH_RETURN(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS_WITH_RETURN(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS_WITH_RETURN(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if (idt != DT_F16)
        CHECK_STATUS_WITH_RETURN(NOT_MATCH);
    if (odt != DT_F16)
        CHECK_STATUS_WITH_RETURN(NOT_MATCH);
    if (fh != fw)
        CHECK_STATUS_WITH_RETURN(NOT_MATCH);
    if (idf != DF_NCHWC8 || odf != DF_NCHWC8)
        CHECK_STATUS_WITH_RETURN(NOT_MATCH);

    EE ret = SUCCESS;
    switch (fdt) {
        case DT_DOREFA:
            ret = convolution_dorefa(inputDesc, (F16*)input,
                                         filterDesc, (BIN8*)filter,
                                         convDesc,
                                         scaleDesc, (F16*)scale,
                                         biasDesc, (F16*)bias,
                                         tmpBytes, tmp,
                                         outputDesc, (F16*)output,
                                         activationMode, arch);
            break;
        case DT_XNOR:
            ret = convolution_xnor(inputDesc, (F16*)input,
                                         filterDesc, (BIN8*)filter,
                                         convDesc,
                                         scaleDesc, (F16*)scale,
                                         biasDesc, (F16*)bias,
                                         tmpBytes, tmp,
                                         outputDesc, (F16*)output,
                                         activationMode, arch);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}