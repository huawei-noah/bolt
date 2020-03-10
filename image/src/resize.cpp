// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <cstring>
#include "sys.h"
#include "type.h"
#include "tensor_desc.h"
#include "error.h"
#include "image.h"
#include "cpu/general/image_general.h"
#include "cpu/arm/image_arm.h"


// params is a pointer to either the target size or the resize ratios
// When resizeDesc specifies DT_U32, params should point to target sizes (height and width)
// When resizeDesc specifies DT_F32, params should point to resize ratios
EE resize_infer_output_size(TensorDesc inputDesc, ResizeDesc resizeDesc, void* params,
    TensorDesc* outputDesc, U32* outputBytes)
{
    if (nullptr == outputDesc || nullptr == outputBytes) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt;
    DataFormat idf;
    U32 in, ic, ih, iw;
    U32 oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    
    switch(resizeDesc.paramDT) {
        case DT_F32: {
            F32 *scales = (F32*)params;
            oh = ih * scales[0];
            ow = iw * scales[1];
            break;
        }
        case DT_U32: {
            U32 *len = (U32*)params;
            oh = len[0];
            ow = len[1];
            break;
        }
        default: {
            CHECK_STATUS(NOT_SUPPORTED);
            return NOT_SUPPORTED;
        }
    }

    *outputDesc = tensor4df(idt, idf, in, ic, oh, ow);
    *outputBytes = tensorNumBytes(*outputDesc);
    return SUCCESS;
}

EE resize(TensorDesc inputDesc, void* input,
        TensorDesc outputDesc, void* output,
        Arch arch)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    CHECK_REQUIREMENT(in == on && ic == oc);
    
    EE ret = SUCCESS;
    if (ih == oh && iw == ow) {
        memcpy(output, input, tensorNumBytes(inputDesc));
        return ret;
    }

    switch (arch) {
        case CPU_GENERAL:
            ret = resize_bilinear_general(inputDesc, input,
                                      outputDesc, output);
            break;
        case ARM_A55:
            ret = resize_bilinear_arm(inputDesc, input,
                                      outputDesc, output);
            break;
        case ARM_A76:
            ret = resize_bilinear_arm(inputDesc, input,
                                      outputDesc, output);
            break;
        case ARM_V8:
            ret = resize_bilinear_arm(inputDesc, input,
                                      outputDesc, output);
            break;
        default:
            ret = NOT_SUPPORTED;
    }
    return ret;
}