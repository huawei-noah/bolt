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
#ifdef _USE_CPU
#include "cpu/image_cpu.h"
#endif
#ifdef _USE_GENERAL
#include "cpu/general/image_general.h"
#endif
#ifdef _USE_NEON
#include "cpu/arm/image_arm.h"
#endif
#ifdef _USE_GPU
#include "gpu/mali/image_mali.h"
#endif
#ifdef _USE_X86
#include "cpu/x86/image_x86.h"
#endif

static bool is_implicit_nhwc(TensorDesc desc) {
    bool ret = false;
    if (desc.df == DF_NCHW && desc.dims[0] == 3 && desc.dims[1] > 3 && desc.dims[2] > 3) {
        ret = true;
    }
    return ret;
}

// params is a pointer to either the target size or the resize ratios
// When paramDT specifies DT_U32, params should point to target sizes (height and width)
// When paramDT specifies DT_F32, params should point to resize ratios
EE resize_infer_output_size_cpu(TensorDesc inputDesc, ResizeParamSpec p, TensorDesc *outputDesc)
{
    DataType idt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw = 1;
    U32 oh, ow = 1;
    bool nhwc = false;
    if (tensorIs3d(inputDesc)) {
        CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &in, &ic, &ih));
    } else if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        nhwc = is_implicit_nhwc(inputDesc);
        if (nhwc) {
            int t = iw;
            iw = ih;
            ih = ic;
            ic = t;
        }
    } else {
        UNI_ERROR_LOG("can support to resize %d-dim tensor.\n", inputDesc.nDims);
    }

    if (p.zoom_factor > 0) {
        int height_in_eff = ih + p.pad_begin + p.pad_end;
        int width_in_eff = iw + p.pad_begin + p.pad_end;
        float shrink_factor = 1.0;
        oh = (height_in_eff - 1) / shrink_factor + 1;
        ow = (width_in_eff - 1) / shrink_factor + 1;
        oh = oh + (oh - 1) * (p.zoom_factor - 1);
        ow = ow + (ow - 1) * (p.zoom_factor - 1);
    } else if (p.num_sizes > 0) {
        oh = p.sizes[0];
        if (p.num_sizes > 1) {
            ow = p.sizes[1];
        }
    } else {
        oh = ih * p.scales[2];
        if (p.num_scales > 3) {
            ow = iw * p.scales[3];
        }
    }
    if (ic % 8 == 0) {
        odf = DF_NCHWC8;
    } else {
        odf = idf;
    }
#ifdef _USE_INT8
    if (idf == DF_NCHWC16) {
        odf = idf;
    }
#endif
    if (tensorIs3d(inputDesc)) {
        *outputDesc = tensor3df(idt, odf, in, ic, oh);
    } else if (tensorIs4d(inputDesc)) {
        if (nhwc) {
            *outputDesc = tensor4df(idt, odf, in, oh, ow, ic);
        } else {
            *outputDesc = tensor4df(idt, odf, in, ic, oh, ow);
        }
    }
    return SUCCESS;
}

EE resize_infer_output_size(
    Tensor *inputTensor, ResizeParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo)
{
    if (inputTensor == nullptr || outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensor->get_desc();
    TensorDesc outputDesc = outputTensor->get_desc();
    EE ret = resize_infer_output_size_cpu(inputDesc, p, &outputDesc);
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        outputDesc.df = inputDesc.df;
#endif
    }
    outputTensor->resize(outputDesc);
    return ret;
}

EE resize_infer_forward_tmp_bytes(
    Tensor inputTensor, ResizeParamSpec p, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo)
{
    if (bytes == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensor.get_desc();
    TensorDesc outputDesc = outputTensor.get_desc();
    *bytes = 0;
    auto arch = archInfo->arch;
    if (IS_GPU(arch)) {
        if (inputDesc.df == DF_NCHW && inputTensor.get_mem_type() != OCLMem) {
            *bytes = tensorNumBytes(inputDesc);
        }
    } else {
        if (DF_NCHW == inputDesc.df && (IS_ARM(arch) || IS_X86(arch))) {
            int channelAxis = inputDesc.nDims - 2;
            U32 paddedC = (inputDesc.dims[channelAxis] + 7) / 8 * 8;
            inputDesc.dims[channelAxis] = paddedC;
            outputDesc.dims[channelAxis] = paddedC;
            *bytes = tensorNumBytes(inputDesc) + tensorNumBytes(outputDesc);
        }
    }
    return SUCCESS;
}

EE resize_bilinear(TensorDesc inputDesc,
    void *input,
    ResizeParamSpec p,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    EE ret = NOT_SUPPORTED;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = resize_bilinear_general(inputDesc, input, p, outputDesc, output);
#endif
#ifdef _USE_X86
    } else if (IS_X86(arch)) {
        ret = resize_bilinear_x86(inputDesc, input, p, tmp, outputDesc, output);
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
        ret = resize_bilinear_arm(inDescARM, inputARM, p, outDescARM, outputARM);
        if (DF_NCHWC8 != outputDesc.df) {
            transformToNCHW(outDescARM, outputARM, outputDesc, output);
        }
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        ret = resize_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc, (GCLMem_t)input, p,
            outputDesc, (GCLMem_t)tmp, (GCLMem_t)output);
#endif
    }
    return ret;
}

EE resize_nearest(TensorDesc inputDesc,
    void *input,
    ResizeParamSpec p,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    EE ret = NOT_SUPPORTED;
    if (IS_X86(arch)) {
#ifdef _USE_X86
        ret = resize_nearest_x86(inputDesc, input, p, outputDesc, output);
#endif
#ifdef _USE_CPU
    } else if (IS_CPU(arch)) {
        ret = resize_nearest_cpu(inputDesc, input, p, outputDesc, output);
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        ret = resize_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc, (GCLMem_t)input, p,
            outputDesc, (GCLMem_t)tmp, (GCLMem_t)output);
#endif
    }
    return ret;
}

static bool update(TensorDesc &inputDesc, TensorDesc &outputDesc) {
    bool ret = false;
    if (is_implicit_nhwc(inputDesc) && inputDesc.dims[0] == outputDesc.dims[0]) {
        TensorDesc desc0 = inputDesc;
        U32 v = inputDesc.dims[0];
        for (U32 i = 0; i < inputDesc.nDims; i++) {
            inputDesc.dims[i - 1] = inputDesc.dims[i];
        }
        inputDesc.dims[inputDesc.nDims - 2] = v;
        inputDesc.df = DF_NCHW;

        TensorDesc desc1 = outputDesc;
        v = outputDesc.dims[0];
        for (U32 i = 1; i < outputDesc.nDims; i++) {
            outputDesc.dims[i - 1] = outputDesc.dims[i];
        }
        outputDesc.dims[outputDesc.nDims - 2] = v;
        outputDesc.df = DF_NHWC;

        UNI_DEBUG_LOG("change input from %s -> %s.\n", tensorDesc2Str(desc0).c_str(),
            tensorDesc2Str(inputDesc).c_str());
        UNI_DEBUG_LOG("change output from %s -> %s.\n", tensorDesc2Str(desc1).c_str(),
            tensorDesc2Str(outputDesc).c_str());
        ret = true;
    }
    return ret;
}

EE resize(
    Tensor inputTensor, ResizeParamSpec p, Tensor tmpTensor, Tensor outputTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);
    void *tmp = get_ptr_from_tensor(tmpTensor, arch);
    update(inputDesc, outputDesc);

    if (inputDesc.nDims == 3) {
        for (int i = inputDesc.nDims; i > 0; i--) {
            inputDesc.dims[i] = inputDesc.dims[i - 1];
            outputDesc.dims[i] = outputDesc.dims[i - 1];
        }
        inputDesc.nDims++;
        outputDesc.nDims++;
    }
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    CHECK_REQUIREMENT(in == on && ic == oc);

    if (ih == oh && iw == ow && IS_CPU(arch)) {
        UNI_MEMCPY(output, input, tensorNumBytes(inputDesc));
        return SUCCESS;
    }

    EE ret = NOT_SUPPORTED;
    switch (p.mode) {
        case RESIZE_NEAREST:
            ret = resize_nearest(inputDesc, input, p, tmp, outputDesc, output, archInfo);
            break;
        case RESIZE_LINEAR:
            ret = resize_bilinear(inputDesc, input, p, tmp, outputDesc, output, archInfo);
            break;
        default:
            break;
    }
    return ret;
}
