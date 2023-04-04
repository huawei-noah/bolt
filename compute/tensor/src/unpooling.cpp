// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "tensor_computing.h"
#ifdef _USE_GENERAL
#include "cpu/general/tensor_computing_general.h"
#endif
#ifdef _USE_NEON
#include "cpu/arm/tensor_computing_arm.h"
#endif
#ifdef _USE_GPU
#include "gpu/mali/tensor_computing_mali.h"
#endif
#ifdef _USE_X86
#include "cpu/x86/tensor_computing_x86.h"
#endif

static inline PoolingParamSpec update_param(TensorDesc inDesc, PoolingParamSpec poolingParamSpec)
{
    if (0 == poolingParamSpec.kernel_w) {
        if (inDesc.nDims > 3) {
            poolingParamSpec.kernel_w = inDesc.dims[0];
        } else {
            poolingParamSpec.kernel_w = 1;
        }
    }
    if (0 == poolingParamSpec.kernel_h) {
        if (inDesc.nDims > 3) {
            poolingParamSpec.kernel_h = inDesc.dims[1];
        } else {
            poolingParamSpec.kernel_h = inDesc.dims[0];
        }
    }
    if (0 == poolingParamSpec.kernel_t) {
        if (inDesc.nDims > 4) {
            poolingParamSpec.kernel_t = inDesc.dims[2];
        } else {
            poolingParamSpec.kernel_t = 1;
        }
    }
    return poolingParamSpec;
}

EE unpooling_infer_output_size(
    Tensor *inputTensor, PoolingParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo)
{
    if (inputTensor == nullptr || outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensor->get_desc();
    TensorDesc newInputDesc = transformDescTo4d(inputDesc);
    TensorDesc outputDesc = outputTensor->get_desc();
    p = update_param(newInputDesc, p);

    DataType idt;
    DataFormat idf;
    U32 in, ic, it, ih, iw;
    if (tensorIs3d(inputDesc)) {
        CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &in, &ic, &ih));
        it = iw = 1;
    } else if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        it = 1;
    } else if (tensorIs5d(inputDesc)) {
        CHECK_STATUS(tensor5dGet(inputDesc, &idt, &idf, &in, &ic, &it, &ih, &iw));
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    U32 ot = 0, oh = 0, ow = 0;
    ot = (it - 1) * p.stride_t + p.kernel_t + p.pad_before + p.pad_after;
    oh = (ih - 1) * p.stride_h + p.kernel_h + p.pad_bottom + p.pad_top;
    ow = (iw - 1) * p.stride_w + p.kernel_w + p.pad_left + p.pad_right;

    DataFormat odf = idf;
    EE ret = SUCCESS;
    if (tensorIs3d(inputDesc)) {
        outputDesc = tensor3df(idt, odf, in, ic, oh);
    } else if (tensorIs4d(inputDesc)) {
        outputDesc = tensor4df(idt, odf, in, ic, oh, ow);
    } else if (tensorIs5d(inputDesc)) {
        outputDesc = tensor5df(idt, odf, in, ic, ot, oh, ow);
    } else {
        ret = NOT_SUPPORTED;
    }
    outputTensor->resize(outputDesc);
    return ret;
}

EE unpooling(std::vector<Tensor> inTensors,
    PoolingParamSpec poolingParamSpec,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    TensorDesc inputDesc = inTensors[0].get_desc();
    F32 *input = (F32 *)get_ptr_from_tensor(inTensors[0], archInfo->arch);
    TensorDesc idxDesc = inTensors[1].get_desc();
    I32 *idx = (I32 *)get_ptr_from_tensor(inTensors[1], archInfo->arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    F32 *output = (F32 *)get_ptr_from_tensor(outputTensor, archInfo->arch);
    void *tmp = get_ptr_from_tensor(tmpTensor, archInfo->arch);
    F32 *realInput = input;
    I32 *realIdx = idx;

    DataFormat dstF = outputDesc.df;
    if (idxDesc.df != dstF) {
        TensorDesc desc = idxDesc;
        desc.df = dstF;
        realIdx = (I32 *)tmp;
        tmp = (U8 *)tmp + tensorNumBytes(idxDesc);
        transformFormat(idxDesc, idx, desc, realIdx);
        idxDesc = desc;
    }
    if (inputDesc.df != dstF) {
        TensorDesc desc = inputDesc;
        desc.df = dstF;
        realInput = (F32 *)tmp;
        tmp = (U8 *)tmp + tensorNumBytes(inputDesc);
        transformFormat(inputDesc, input, desc, realInput);
        inputDesc = desc;
    }

    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    U32 strideH = poolingParamSpec.stride_h;
    U32 strideW = poolingParamSpec.stride_w;
    U32 paddingT = poolingParamSpec.pad_top;
    U32 paddingL = poolingParamSpec.pad_left;
    U32 kernelSizeH = poolingParamSpec.kernel_h;
    U32 kernelSizeW = poolingParamSpec.kernel_w;

    UNI_MEMSET(output, 0, tensorNumBytes(outputDesc));

    EE ret = SUCCESS;
    if (dstF == DF_NCHWC8) {
        ic /= 8;
        U32 ohow = oh * ow;
        U32 ocTile = oc * ohow;
        U32 loops = in * ic * ih * iw;
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
        for (U32 n = 0; n < loops; ++n) {
            for (U32 i = 0; i < 8; ++i) {
                U32 iIdx = n * 8 + i;
                I32 rawIdx = realIdx[iIdx];
                U32 ocIdx = rawIdx % ocTile / ohow;
                U32 ohIdx = rawIdx % ohow / ow;
                U32 owIdx = rawIdx % ow;
                U32 oIdx = rawIdx / ocTile * ocTile + (ocIdx / 8) * ohow * 8 + ohIdx * ow * 8 +
                    owIdx * 8 + (ocIdx % 8);
                output[oIdx] = realInput[iIdx];
            }
        }
    } else {
        ret = NOT_SUPPORTED;
    }

    return ret;
}

EE unpooling_infer_forward_tmp_bytes(
    std::vector<Tensor> inTensors, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo)
{
    if (inTensors.size() < 2) {
        CHECK_STATUS(NOT_MATCH);
    }
    DataFormat outputDf = outputTensor.get_desc().df;
    TensorDesc inputDesc = inTensors[0].get_desc();
    TensorDesc idxDesc = inTensors[1].get_desc();
    DataFormat inputDf = inputDesc.df;
    DataFormat idxDf = idxDesc.df;

    *bytes = 0;
    if (inputDf != outputDf) {
        *bytes += tensorNumBytes(inputDesc);
    }
    if (idxDf != outputDf) {
        *bytes += tensorNumBytes(idxDesc);
    }

    return SUCCESS;
}
