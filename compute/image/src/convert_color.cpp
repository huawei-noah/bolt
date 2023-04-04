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
#ifdef _USE_GPU
#include "gpu/mali/image_mali.h"
#endif

EE convert_color_infer_output_size(
    Tensor *inputTensor, ConvertColorParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo)
{
    if (inputTensor == nullptr || outputTensor == nullptr) {
        return NULL_POINTER;
    }
    TensorDesc inputDesc = inputTensor->get_desc();
    if (inputDesc.nDims != 4) {
        return NOT_SUPPORTED;
    }
    DataType idt, odt;
    DataFormat idf;
    U32 in, ic, ih, iw;
    U32 oc, oh, ow;
    if (inputDesc.df == DF_NHWC) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    } else {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ih, &iw, &ic));
    }
    odt = p.dt;
    if (p.src == YUV_NV21) {
        CHECK_REQUIREMENT(ic == 1);
        oh = ih / 3 * 2;
        ow = iw;
        if (p.dst == RGB_0_255 || p.dst == RGB_0_1 || p.dst == BGR_0_255 || p.dst == BGR_0_1) {
            oc = 3;
        } else if (p.dst == RGBA_0_255 || p.dst == RGBA_0_1 || p.dst == BGRA_0_255 ||
            p.dst == BGRA_0_1) {
            oc = 4;
        } else {
            return NOT_SUPPORTED;
        }
    } else if (p.src == RGB_0_255 || p.src == RGB_0_1 || p.src == BGR_0_255 || p.src == BGR_0_1 ||
        p.src == RGBA_0_255 || p.src == RGBA_0_1 || p.src == BGRA_0_255 || p.src == BGR_0_1) {
        if (p.src == RGB_0_255 || p.src == RGB_0_1 || p.src == BGR_0_255 || p.src == BGR_0_1) {
            CHECK_REQUIREMENT(ic == 3);
        }
        if (p.src == RGBA_0_255 || p.src == RGBA_0_1 || p.src == BGRA_0_255 || p.src == BGRA_0_1) {
            CHECK_REQUIREMENT(ic == 4);
        }
        if (p.dst == YUV_NV21) {
            odt = DT_U8;
            oh = ih / 2 * 3;
            ow = iw;
            oc = 1;
        } else {
            return NOT_SUPPORTED;
        }
    } else {
        return NOT_SUPPORTED;
    }
    TensorDesc outputDesc;
    if (inputDesc.df == DF_NHWC) {
        outputDesc = tensor4df(odt, idf, in, oc, oh, ow);
    } else {
        outputDesc = tensor4df(odt, idf, in, oh, ow, oc);
    }
    outputTensor->resize(outputDesc);
    return SUCCESS;
}

EE convert_color(
    Tensor inputTensor, ConvertColorParamSpec p, Tensor outputTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
#ifdef _USE_CPU
        ret = convert_color_cpu(inputDesc, input, p, outputDesc, output);
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        ret = convert_color_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc,
            (GCLMem_t)input, p, outputDesc, (GCLMem_t)output);
#endif
    }
    return ret;
}
