// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <vector>
#include <set>
#include "cpu/tensor_computing_cpu.h"
#ifdef _USE_GENERAL
#include "cpu/general/tensor_computing_general.h"
#endif
#ifdef _USE_NEON
#include "cpu/arm/tensor_computing_arm.h"
#endif
#ifdef _USE_X86
#include "cpu/x86/tensor_computing_x86.h"
#endif
#include "tensor_transpose.h"

static std::vector<U32> calculateRelativeLocalIndex_cpu(U32 *indexes, U32 *dims, U32 nDims)
{
    std::vector<U32> relativeIndexes(nDims);
    for (U32 i = 0; i < nDims; i++) {
        relativeIndexes[i] = indexes[i] % dims[i];
    }
    return relativeIndexes;
}

static void get_dim_nonone_bound(TensorDesc desc, int *left, int *right)
{
    *left = -1;
    for (U32 i = 0; i < desc.nDims; i++) {
        if (desc.dims[i] == 1) {
            *left = i;
        } else {
            break;
        }
    }
    *right = desc.nDims;
    for (I32 i = desc.nDims - 1; i >= 0; i--) {
        if (desc.dims[i] == 1) {
            *right = i;
        } else {
            break;
        }
    }
    *left = *left + 1;
    *right = *right - 1;
}

static int scale_axis(
    std::vector<TensorDesc> inputDesc, TensorDesc outputDesc, int *scaleId, TensorDesc *scaleDesc)
{
    if (inputDesc.size() != 2) {
        return -1;
    }
    int al, ar, bl, br;
    get_dim_nonone_bound(inputDesc[0], &al, &ar);
    get_dim_nonone_bound(inputDesc[1], &bl, &br);
    // use power operator
    if (al > ar) {
        return -2;
    }
    if (bl > br) {
        return -3;
    }
    int cl = UNI_MIN(al, bl);
    int cr = UNI_MAX(ar, br);
    int alpha = -1;
    if (cr - cl > ar - al) {
        alpha = 0;
    }
    if (cr - cl > br - bl) {
        alpha = 1;
    }
    if (alpha < 0) {
        return -1;
    }
    int dl = UNI_MAX(al, bl);
    int dr = UNI_MIN(ar, br);
    for (int i = dl; i <= dr; i++) {
        if (inputDesc[0].dims[i] != inputDesc[1].dims[i]) {
            return -1;
        }
    }
    int axis = cr - dr;
    *scaleId = 1 - alpha;
    *scaleDesc = inputDesc[*scaleId];
    scaleDesc->nDims = (dl - cl) + (cr - dr) + 1;
    int j = 0;
    for (int i = cl; i < dl; i++) {
        scaleDesc->dims[j++] = inputDesc[*scaleId].dims[i];
    }
    scaleDesc->dims[j] = 1;
    for (int i = dl; i <= dr; i++) {
        scaleDesc->dims[j] *= inputDesc[*scaleId].dims[i];
    }
    for (int i = dr + 1; i <= cr; i++) {
        scaleDesc->dims[++j] = inputDesc[*scaleId].dims[i];
    }
    if (dr == cr) {
        scaleDesc->dims[++j] = 1;
        scaleDesc->nDims++;
        axis++;
    }
    return axis;
}

static void align_param(
    std::vector<TensorDesc> &inputDesc, std::vector<void *> &input, void *tmp, TensorDesc &outputDesc)
{
    U32 num = inputDesc.size();
    U8 *ptr = (U8 *)tmp;
    std::set<DataFormat> nchw = {DF_NORMAL, DF_MTK, DF_MKT, DF_NCHW};
    for (U32 i = 0; i < num; i++) {
        if (inputDesc[i].nDims <= 2 ||
            (nchw.find(inputDesc[i].df) != nchw.end() && nchw.find(outputDesc.df) != nchw.end())) {
            continue;
        }
        if (inputDesc[i].df != outputDesc.df ||
            tensorNumElements(inputDesc[i]) != tensorNumElements(outputDesc)) {
            // Kaldi tdnn special case
            if (inputDesc[i].df == DF_NHWC && inputDesc[i].nDims == 3) {
                inputDesc[i] = tensor4df(inputDesc[i].dt, DF_NHWC, inputDesc[i].dims[2],
                    inputDesc[i].dims[0], inputDesc[i].dims[1], 1);
            }
            TensorDesc tmpDesc = outputDesc;
            if (tensorNumElements(inputDesc[i]) < tensorNumElements(outputDesc)) {
                tmpDesc = inputDesc[i];
                tmpDesc.df = outputDesc.df;
            }
            CHECK_STATUS(transformFormat(inputDesc[i], input[i], tmpDesc, ptr));
            inputDesc[i] = tmpDesc;
            input[i] = ptr;
            ptr += tensorNumBytes(tmpDesc);
        }
    }

    I32 oneCount = 0;
    for (int i = 0; i < (int)outputDesc.nDims - 1; i++) {
        if (outputDesc.dims[i] == 1) {
            oneCount++;
        } else {
            break;
        }
    }

    for (int i = 0; i < (int)outputDesc.nDims - oneCount; i++) {
        outputDesc.dims[i] = outputDesc.dims[oneCount + i];
    }
    outputDesc.nDims = outputDesc.nDims - oneCount;

    for (U32 i = 0; i < num; i++) {
        TensorDesc desc = inputDesc[i];
        for (int j = 0; j < (int)inputDesc[i].nDims - oneCount; j++) {
            desc.dims[j] = inputDesc[i].dims[oneCount + j];
        }
        desc.nDims = inputDesc[i].nDims - oneCount;
        for (U32 j = desc.nDims; j < outputDesc.nDims; j++) {
            desc.dims[j] = 1;
        }
        desc.nDims = outputDesc.nDims;
        inputDesc[i] = desc;
    }
}

static EE eltwise_kernel(std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    EltwiseParamSpec p,
    TensorDesc outputDesc,
    void *output,
    Arch arch)
{
    U32 num = inputDesc.size();
    int lastDimSize = outputDesc.dims[0];
    std::vector<int> lastDimSizes(num);
    bool sameDim = true;
    for (U32 i = 0; i < num; i++) {
        lastDimSizes[i] = inputDesc[i].dims[0];
        if (lastDimSizes[i] != lastDimSize) {
            sameDim = false;
            if (inputDesc[0].df == DF_NCHWC8 || inputDesc[0].df == DF_NCHWC16) {
                UNI_ERROR_LOG("For NCHWC8 and NCHWC16, eltwise can only handle inputs with "
                              "matching widths\n");
            }
        }
    }
    for (U32 i = 1; i < outputDesc.nDims; i++) {
        for (U32 j = 0; j < num; j++) {
            if (inputDesc[j].dims[i] != outputDesc.dims[i]) {
                sameDim = false;
                break;
            }
        }
        if (sameDim) {
            lastDimSize *= outputDesc.dims[i];
            for (U32 j = 0; j < num; j++) {
                lastDimSizes[j] *= inputDesc[j].dims[i];
            }
        } else {
            break;
        }
    }

    EE ret = NOT_SUPPORTED;
    if (sameDim) {  // if merged to the next loop, it will be slower when using openmp.
        if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
            ret = eltwise_general(outputDesc.dt, input, lastDimSizes, num, lastDimSize, output, p.mode);
#endif
#ifdef _USE_NEON
        } else if (IS_ARM(arch)) {
            ret = eltwise_arm(outputDesc.dt, input, lastDimSizes, num, lastDimSize, output, p.mode);
#endif
#ifdef _USE_X86
        } else if (IS_X86(arch)) {
            ret = eltwise_x86(outputDesc.dt, input, lastDimSizes, num, lastDimSize, output, p.mode);
#endif
        }
        return ret;
    }

    U32 loopNum = tensorNumElements(outputDesc) / lastDimSize;
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
    for (U32 i = 0; i < loopNum; ++i) {
        std::vector<U32> index = calculateLocalIndex(i * lastDimSize, outputDesc.dims, outputDesc.nDims);
        std::vector<void *> ip(num);
        for (U32 j = 0; j < num; j++) {
            std::vector<U32> relativeIndex = calculateRelativeLocalIndex_cpu(
                index.data(), inputDesc[j].dims, inputDesc[j].nDims);
            U32 globalIndex =
                calculateGlobalIndex(relativeIndex.data(), inputDesc[j].dims, inputDesc[j].nDims);
            ip[j] = (U8 *)(input[j]) + globalIndex * bytesOf(inputDesc[j].dt);
        }
        U8 *op = (U8 *)output + i * lastDimSize * bytesOf(outputDesc.dt);
        if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
            ret = eltwise_general(outputDesc.dt, ip, lastDimSizes, num, lastDimSize, op, p.mode);
#endif
#ifdef _USE_NEON
        } else if (IS_ARM(arch)) {
            ret = eltwise_arm(outputDesc.dt, ip, lastDimSizes, num, lastDimSize, op, p.mode);
#endif
#ifdef _USE_X86
        } else if (IS_X86(arch)) {
            ret = eltwise_x86(outputDesc.dt, ip, lastDimSizes, num, lastDimSize, op, p.mode);
#endif
        }
    }
    return ret;
}

// [1, 10, 10] + [1, 10, 10] = [1, 10, 10]
// [1, 10, 1] + [1, 1, 10] = [1, 10, 10]
// [1, 20, 10] + [10] = [1. 20, 10] + [1, 1, 10] = [1, 20, 10]
EE eltwise_cpu(std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    EltwiseParamSpec p,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    Arch arch)
{
    U32 num = inputDesc.size();
    if (num <= 1 || outputDesc.nDims < 1) {
        return NOT_MATCH;
    }
    if (tensorNumElements(outputDesc) == 0) {
        return SUCCESS;
    }
    align_param(inputDesc, input, tmp, outputDesc);

    EE ret = NOT_SUPPORTED;
    int scaleId = -1;
    TensorDesc scaleDesc;
    int axis = scale_axis(inputDesc, outputDesc, &scaleId, &scaleDesc);
    if (axis >= 0 && (p.mode == ELTWISE_PROD || p.mode == ELTWISE_SUM)) {
        ScaleParamSpec sp;
        sp.axis = axis;
        if (p.mode == ELTWISE_PROD) {
            ret = scale_cpu(scaleDesc, input[scaleId], input[1 - scaleId], nullptr, sp, scaleDesc,
                output, arch);
        } else {
            ret = scale_cpu(scaleDesc, input[scaleId], nullptr, input[1 - scaleId], sp, scaleDesc,
                output, arch);
        }
    } else {
        ret = eltwise_kernel(inputDesc, input, p, outputDesc, output, arch);
    }
    if (ret == SUCCESS && p.activation_type != ACTIVATION_NULL) {
        ActivationParamSpec ap;
        ap.mode = p.activation_type;
        ret = activation_cpu(outputDesc, output, ap, outputDesc, output, arch);
    }
    return ret;
}
