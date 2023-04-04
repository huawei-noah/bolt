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
#include "cpu/tensor_computing_cpu.h"
#include "cpu/bcast.h"
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
            ret = eltwise_general(
                outputDesc.dt, input, lastDimSizes, num, lastDimSize, output, p.mode);
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
        std::vector<U32> index =
            calculateLocalIndex(i * lastDimSize, outputDesc.dims, outputDesc.nDims);
        std::vector<void *> ip(num);
        for (U32 j = 0; j < num; j++) {
            std::vector<U32> relativeIndex =
                calculateRelativeLocalIndex(index.data(), inputDesc[j].dims, inputDesc[j].nDims);
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
    tmp = align_param(inputDesc, input, tmpBytes, tmp, outputDesc);

    EE ret = NOT_SUPPORTED;
    int scaleId = -1;
    TensorDesc scaleDesc;
    int axis = scale_axis(inputDesc, outputDesc, &scaleId, &scaleDesc);
    if (axis >= 0 &&
        (p.mode == ELTWISE_PROD || p.mode == ELTWISE_SUM || p.mode == ELTWISE_SUB ||
            p.mode == ELTWISE_DIV)) {
        UNI_DETAIL_LOG("eltwise use scale input:%d %s on axis:%d\n", scaleId,
            tensorDesc2Str(scaleDesc).c_str(), axis);
        ScaleParamSpec sp;
        sp.axis = axis;
        if (p.mode == ELTWISE_PROD) {
            ret = scale_cpu(scaleDesc, input[scaleId], input[1 - scaleId], nullptr, sp, scaleDesc,
                output, arch);
        } else if (p.mode == ELTWISE_SUM) {
            ret = scale_cpu(scaleDesc, input[scaleId], nullptr, input[1 - scaleId], sp, scaleDesc,
                output, arch);
        } else if (p.mode == ELTWISE_SUB || p.mode == ELTWISE_DIV) {
            PowerParamSpec pp;
            void *scale, *bias;
            if (p.mode == ELTWISE_SUB) {
                pp = {-1, 0, 1};
                scale = nullptr;
                bias = tmp;
            } else {
                pp = {1, 0, -1};
                scale = tmp;
                bias = nullptr;
            }
            CHECK_REQUIREMENT(tmpBytes >= tensorNumBytes(inputDesc[1 - scaleId]));
            CHECK_STATUS(power_cpu(
                inputDesc[1 - scaleId], input[1 - scaleId], pp, inputDesc[1 - scaleId], tmp, arch));
            ret = scale_cpu(scaleDesc, input[scaleId], scale, bias, sp, scaleDesc, output, arch);
            if (scaleId == 1) {
                CHECK_STATUS(power_cpu(scaleDesc, output, pp, scaleDesc, output, arch));
            }
        } else {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    } else if (axis == -2 &&
        (p.mode == ELTWISE_PROD || p.mode == ELTWISE_SUM || p.mode == ELTWISE_SUB ||
            p.mode == ELTWISE_DIV)) {
        UNI_DETAIL_LOG("eltwise use power input:%d %s\n", scaleId, tensorDesc2Str(scaleDesc).c_str());
        float value;
        transformToFloat(inputDesc[1 - scaleId].dt, input[1 - scaleId], &value, 1);
        PowerParamSpec pp1, pp2 = {1, 0, 1};
        if (p.mode == ELTWISE_PROD) {
            pp1 = {value, 0, 1};
        } else if (p.mode == ELTWISE_SUM) {
            pp1 = {1, value, 1};
        } else if (p.mode == ELTWISE_SUB) {
            if (scaleId == 0) {
                pp1 = {1, -value, 1};
            } else {
                pp1 = {-1, value, 1};
            }
        } else if (p.mode == ELTWISE_DIV) {
            if (scaleId == 0) {
                pp1 = {1 / value, 0, 1};
            } else {
                pp1 = {1, 0, -1};
                pp2 = {value, 0, 1};
            }
        } else {
            return NOT_SUPPORTED;
        }
        int *a = (int *)output;
        ret = power_cpu(inputDesc[scaleId], input[scaleId], pp1, outputDesc, output, arch);
        if (ret == SUCCESS && pp2.scale != 1) {
            ret = power_cpu(outputDesc, output, pp2, outputDesc, output, arch);
        }
    } else {
        UNI_DETAIL_LOG("eltwise use naive implementation.\n");
        ret = eltwise_kernel(inputDesc, input, p, outputDesc, output, arch);
    }
    if (ret == SUCCESS && p.activation_type != ACTIVATION_NULL) {
        ActivationParamSpec ap;
        ap.mode = p.activation_type;
        ret = activation_cpu(outputDesc, output, ap, outputDesc, output, nullptr, arch);
    }
    return ret;
}
