// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "image.h"
#ifdef _USE_GENERAL
#include "cpu/general/image_general.h"
#endif
#ifdef _USE_NEON
#include "cpu/arm/image_arm.h"
#endif
#ifdef _USE_MALI
#include "gpu/mali/image_mali.h"
#endif
#ifdef _USE_X86
#include "cpu/x86/image_x86.h"
#endif
#include <string.h>

// params is a pointer to either the target size or the resize ratios
// When paramDT specifies DT_U32, params should point to target sizes (height and width)
// When paramDT specifies DT_F32, params should point to resize ratios
EE resize_infer_output_size_cpu(
    TensorDesc inputDesc, DataType paramDT, void *params, TensorDesc *outputDesc, U32 *outputBytes)
{
    if (nullptr == outputDesc || nullptr == outputBytes) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw;
    U32 oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));

    switch (paramDT) {
        case DT_F32: {
            F32 *scales = (F32 *)params;
            oh = ih * scales[0];
            ow = iw * scales[1];
            break;
        }
        case DT_U32: {
            U32 *len = (U32 *)params;
            oh = len[0];
            ow = len[1];
            break;
        }
        default: {
            return NOT_SUPPORTED;
        }
    }
    if (ic % 8 == 0) {
        odf = DF_NCHWC8;
    } else {
        odf = idf;
    }
    *outputDesc = tensor4df(idt, odf, in, ic, oh, ow);
    *outputBytes = tensorNumBytes(*outputDesc);
    return SUCCESS;
}

EE resize_infer_output_size(Tensor *inputTensor,
    DataType paramDT,
    void *params,
    Tensor *outputTensor,
    U32 *outputBytes,
    ArchInfo_t archInfo)
{
    if (inputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensor->get_desc();
    TensorDesc outputDesc = outputTensor->get_desc();
    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        GCLMemDesc gclmemInputDesc = ocl_get_desc(*inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(*outputTensor);
        ret = resize_infer_output_size_mali(inputDesc, paramDT, params, &outputDesc, outputBytes,
            &gclmemInputDesc, &gclmemOutputDesc);
        ocl_set_desc(inputTensor, gclmemInputDesc);
        ocl_set_desc(outputTensor, gclmemOutputDesc);
#endif
    } else {
        ret = resize_infer_output_size_cpu(inputDesc, paramDT, params, &outputDesc, outputBytes);
    }
    outputTensor->resize(outputDesc);
    return ret;
}

EE resize_bilinear(TensorDesc inputDesc,
    void *input,
    TensorDesc outputDesc,
    void *output,
    void *tmp,
    ResizeParamSpec p,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    EE ret = NOT_SUPPORTED;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = resize_bilinear_general(inputDesc, input, outputDesc, output);
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        ret = resize_bilinear_x86(inputDesc, input, outputDesc, tmp, output, p);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        U8 *inputARM = (U8 *)input;
        U8 *outputARM = (U8 *)output;
        TensorDesc inDescARM = inputDesc;
        TensorDesc outDescARM = outputDesc;
        if (DF_NCHWC8 != inputDesc.df) {
            U32 paddedC = (inputDesc.dims[2] + 7) / 8 * 8;
            inDescARM.dims[2] = paddedC;
            inDescARM.df = DF_NCHWC8;
            outDescARM.dims[2] = paddedC;
            outDescARM.df = DF_NCHWC8;
            inputARM = (U8 *)tmp;
            outputARM = inputARM + tensorNumBytes(inDescARM);
            transformNCHWToNCHWC8(inputDesc, input, inDescARM, inputARM);
        }
        ret = resize_bilinear_arm(inDescARM, inputARM, outDescARM, outputARM);
        if (DF_NCHWC8 != outputDesc.df) {
            transformToNCHW(outDescARM, outputARM, outputDesc, output);
        }
#endif
#ifdef _USE_MALI
    } else if (IS_MALI_GPU(arch)) {
        ret = resize_bilinear_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc,
            (GCLMem_t)input, outputDesc, (GCLMem_t)output);
#endif
    }
    CHECK_STATUS(ret);
    return ret;
}

EE resize_nearest(TensorDesc inputDesc,
    void *input,
    TensorDesc outputDesc,
    void *output,
    void *tmp,
    ResizeParamSpec p,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
#ifdef _USE_CPU
        ret = resize_nearest_general(inputDesc, input, outputDesc, output);
#endif
    }
    CHECK_STATUS(ret);
    return ret;
}

EE resize(
    Tensor inputTensor, Tensor tmpTensor, Tensor outputTensor, ResizeParamSpec p, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);
    void *tmp = get_ptr_from_tensor(tmpTensor, arch);

    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    CHECK_REQUIREMENT(in == on && ic == oc);

    if (ih == oh && iw == ow && archInfo->arch != MALI) {
        memcpy(output, input, tensorNumBytes(inputDesc));
        return SUCCESS;
    }

    EE ret = SUCCESS;
    switch (p.mode) {
        case NEAREST:
            ret = resize_nearest(inputDesc, input, outputDesc, output, tmp, p, archInfo);
            break;
        case LINEAR:
        default:
            ret = resize_bilinear(inputDesc, input, outputDesc, output, tmp, p, archInfo);
            break;
    }
    return ret;
}
