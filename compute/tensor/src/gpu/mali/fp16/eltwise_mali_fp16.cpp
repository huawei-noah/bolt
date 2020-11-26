// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include <algorithm>
#include <unordered_set>

#include "sys.h"
#include "error.h"
#include "types.h"
#include "gpu/mali/fp16/eltwise_mali_fp16.h"

bool eltwise_same_desc(std::vector<TensorDesc> inputDesc, U32 *arrayDimMax)
{
    U32 size = inputDesc.size();
    U32 dimMax = 0;
    for (U32 i = 1; i < size; i++) {
        if (inputDesc[i].nDims > inputDesc[dimMax].nDims) {
            dimMax = i;
        } else if (inputDesc[i].nDims == inputDesc[dimMax].nDims) {
            U32 nDims = inputDesc[dimMax].nDims;
            U32 sign[8];
            if (nDims > 8) {
                CHECK_STATUS(NOT_SUPPORTED);
            }
            for (U32 j = 0; j < nDims; j++) {
                if (inputDesc[i].dims[j] > inputDesc[dimMax].dims[j]) {
                    sign[j] = 2;
                } else if (inputDesc[i].dims[j] == inputDesc[dimMax].dims[j]) {
                    sign[j] = 1;
                } else {
                    sign[j] = 0;
                }
            }
            if (*std::max_element(sign, sign + nDims) == 2 &&
                *std::min_element(sign, sign + nDims) == 1) {
                dimMax = i;
            }
            if (*std::max_element(sign, sign + nDims) == 2 &&
                *std::min_element(sign, sign + nDims) == 0) {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        }
    }

    bool sameDesc = true;
    DataFormat idf;
    U32 in, ic, ih, iw, it;
    tensorSelectGet(inputDesc[0], NULL, &idf, &in, &ic, &ih, &iw, &it);
    for (U32 i = 1; i < size; i++) {
        DataFormat tdf;
        U32 tn, tc, th, tw, tt;
        tensorSelectGet(inputDesc[i], NULL, &tdf, &tn, &tc, &th, &tw, &tt);
        if (tdf != idf || in != tn || ic != tc || ih != th || iw != tw || it != tt) {
            sameDesc = false;
            break;
        }
    }
    *arrayDimMax = dimMax;
    return sameDesc;
}

inline EE eltwise_checkpara_mali_fp16(
    std::vector<TensorDesc> inputDesc, std::vector<void *> input, TensorDesc outputDesc)
{
    for (auto it : inputDesc) {
        if (it.dt != outputDesc.dt) {
            return NOT_SUPPORTED;
        }
    }
    if (outputDesc.dt != DT_F16) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE eltwise_core_mali_fp16(GCLHandle_t handle,
    std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    TensorDesc outputDesc,
    GCLMem_t output,
    EltwiseParamSpec eltwiseDesc)
{
    UNUSED(outputDesc);
    U32 iw, ih, ic, in, it;
    U32 arrayDimMax;
    bool sameDesc = eltwise_same_desc(inputDesc, &arrayDimMax);
    tensorSelectGet(inputDesc[arrayDimMax], NULL, NULL, &in, &ic, &ih, &iw, &it);

    U32 num = input.size();
    std::vector<GCLMem_t> inputMem;
    for (U32 i = 0; i < num; ++i) {
        inputMem.push_back((GCLMem_t)input[i]);
    }
    cl_mem outbuf;
    outbuf = output->mem;

    U32 ow_str, oh_str, oc_str, ow_off, oh_off;
    std::vector<U32> iw_str;
    std::vector<U32> ih_str;
    std::vector<U32> iw_off;
    std::vector<U32> ih_off;
    for (U32 i = 0; i < num; ++i) {
        U32 w_str, h_str, w_off, h_off;
        get_gclmem_dim(inputMem[i]->desc, &w_str, &h_str, NULL, &w_off, &h_off);
        iw_str.push_back(w_str);
        ih_str.push_back(h_str);
        iw_off.push_back(w_off);
        ih_off.push_back(h_off);
    }
    get_gclmem_dim(output->desc, &ow_str, &oh_str, &oc_str, &ow_off, &oh_off);

    char modeName[16];
    char activeName[16];
    char kernelName[128];
    EltwiseMode eltwiseMode = eltwiseDesc.elt_mode;
    ActivationMode activeMode = eltwiseDesc.activation_type;

    Kernel kernel;
    if (eltwiseMode == ELTWISE_MAX) {
        strcpy(modeName, "max");
    }
    if (eltwiseMode == ELTWISE_SUM) {
        strcpy(modeName, "sum");
    }
    if (eltwiseMode == ELTWISE_PROD) {
        strcpy(modeName, "prod");
    }
    switch (activeMode) {
        case ACTIVATION_RELU:
            strcpy(activeName, "relu_");
            break;
        case ACTIVATION_NULL:
            strcpy(activeName, "");
            break;
        default:
            return NOT_SUPPORTED;
    }
    U32 gs[3] = {ih, iw, (ic + 3) / 4 * in * it};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    if (activeMode != ACTIVATION_NULL && !sameDesc) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    if (sameDesc) {
        char formatName[16] = "";
        ic = ic * in * it;
        if (inputMem[0]->desc.memFormat == DF_NCHW) {
            strcpy(formatName, "nchw_");
            gs[0] = (iw + 3) / 4;
            gs[1] = ih;
            gs[2] = ic;
            if (output->desc.memFormat == DF_NCWHC4) {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        }
        sprintf(kernelName, "eltwise_%s%s%s%d", formatName, activeName, modeName, num);
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
        switch (num) {
            case 1:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ic, oh_str, ow_str, oh_off, ow_off,
                    gs[0], gs[1], ih_str[0], iw_str[0], ih_off[0], iw_off[0], inputMem[0]->mem,
                    outbuf));
                break;
            case 2:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ic, oh_str, ow_str, oh_off, ow_off,
                    gs[0], gs[1], ih_str[0], iw_str[0], ih_off[0], iw_off[0], inputMem[0]->mem,
                    ih_str[1], iw_str[1], ih_off[1], iw_off[1], inputMem[1]->mem, outbuf));
                break;
            case 3:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ic, oh_str, ow_str, oh_off, ow_off,
                    gs[0], gs[1], ih_str[0], iw_str[0], ih_off[0], iw_off[0], inputMem[0]->mem,
                    ih_str[1], iw_str[1], ih_off[1], iw_off[1], inputMem[1]->mem, ih_str[2],
                    iw_str[2], ih_off[2], iw_off[2], inputMem[2]->mem, outbuf));
                break;
            case 4:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ic, oh_str, ow_str, oh_off, ow_off,
                    gs[0], gs[1], ih_str[0], iw_str[0], ih_off[0], iw_off[0], inputMem[0]->mem,
                    ih_str[1], iw_str[1], ih_off[1], iw_off[1], inputMem[1]->mem, ih_str[2],
                    iw_str[2], ih_off[2], iw_off[2], inputMem[2]->mem, ih_str[3], iw_str[3],
                    ih_off[3], iw_off[3], inputMem[3]->mem, outbuf));
                break;
            default:
                return NOT_SUPPORTED;
        }
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
        handle->t_total += handle->t_execute;
#endif
        return SUCCESS;
    } else {
        if (num > 2) {
            CHECK_STATUS(NOT_SUPPORTED)
        }
        DataFormat mf[2];
        mf[0] = inputMem[arrayDimMax]->desc.memFormat;
        mf[1] = inputMem[1 - arrayDimMax]->desc.memFormat;
        if (mf[0] == DF_NCWHC4) {
            U32 w_str, h_str, c_str, w_off, h_off;
            get_gclmem_dim(inputMem[1 - arrayDimMax]->desc, &w_str, &h_str, &c_str, &w_off, &h_off);
            U32 ic_non_max;
            tensorSelectGet(inputDesc[1 - arrayDimMax], NULL, NULL, NULL, &ic_non_max, NULL, NULL);
            char formatName[16] = "";
            if (mf[1] == DF_NCHW) {
                strcpy(formatName, "nchw_");
            }
            char broadAxis[16] = "";
            if (w_str == 1 && h_str == 1 && c_str == 1) {
                strcpy(broadAxis, "_xyz");
            } else if (w_str == 1 && h_str == 1) {
                strcpy(broadAxis, "_xy");
            } else if (w_str != 1 && h_str == 1) {
                strcpy(broadAxis, "_y");
            } else if (w_str == 1 && h_str != 1) {
                strcpy(broadAxis, "_x");
            } else {
                CHECK_STATUS(NOT_SUPPORTED);
            }
            sprintf(kernelName, "eltwise_broadcast_%s%s%s", formatName, broadAxis, modeName);
            CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, ic_non_max, ih_str[arrayDimMax],
                iw_str[arrayDimMax], ih_off[arrayDimMax], iw_off[arrayDimMax], oh_str, ow_str,
                oh_off, ow_off, inputMem[arrayDimMax]->mem, inputMem[1 - arrayDimMax]->mem, outbuf));
            gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
            handle->t_total += handle->t_execute;
#endif
            return SUCCESS;
        }
    }
    return NOT_SUPPORTED;
}

EE eltwise_mali_fp16(GCLHandle_t handle,
    std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    TensorDesc outputDesc,
    GCLMem_t output,
    EltwiseParamSpec eltwiseDesc)
{
    CHECK_STATUS(eltwise_checkpara_mali_fp16(inputDesc, input, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(eltwise_core_mali_fp16(handle, inputDesc, input, outputDesc, output, eltwiseDesc));
    return SUCCESS;
}
