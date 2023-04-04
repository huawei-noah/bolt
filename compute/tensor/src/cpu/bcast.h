// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _BCAST_MODE_H
#define _BCAST_MODE_H

#include <vector>
#include "tensor_transpose.h"

inline void get_dim_nonone_bound(TensorDesc desc, int *left, int *right)
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

inline int scale_axis(
    std::vector<TensorDesc> inputDesc, TensorDesc outputDesc, int *scaleId, TensorDesc *scaleDesc)
{
    *scaleId = -1;
    if (inputDesc.size() != 2) {
        return -1;
    }
    int al, ar, bl, br;
    get_dim_nonone_bound(inputDesc[0], &al, &ar);
    get_dim_nonone_bound(inputDesc[1], &bl, &br);
    // use power operator
    if (al > ar) {
        *scaleId = 1;
        return -2;
    }
    if (bl > br) {
        *scaleId = 0;
        return -2;
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
    if (dl > dr) {
        return -1;
    }
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

inline void *align_param(std::vector<TensorDesc> &inputDesc,
    std::vector<void *> &input,
    U32 &tmpBytes,
    void *tmp,
    TensorDesc &outputDesc)
{
    U32 num = inputDesc.size();
    U8 *ptr = (U8 *)tmp;
    for (U32 i = 0; i < num; i++) {
        if (input.size() > 0 && inputDesc[i].dt != outputDesc.dt) {
            TensorDesc desc = inputDesc[i];
            desc.dt = outputDesc.dt;
            if (tmpBytes >= tensorNumBytes(desc)) {
                tmpBytes -= tensorNumBytes(desc);
                U32 len = tensorNumElements(desc);
                if (inputDesc[i].dt == DT_F32) {
                    transformFromFloat(outputDesc.dt, (const float*)input[i], ptr, len);
                }
                if (outputDesc.dt == DT_F32) {
                    transformToFloat(inputDesc[i].dt, input[i], (float *)ptr, len);
                }
                input[i] = ptr;
                ptr += tensorNumBytes(desc);
                inputDesc[i] = desc;
            }
        }
        if (inputDesc[i].nDims <= 2 || isSameDataFormat(inputDesc[i].df, outputDesc.df)) {
            continue;
        }
        // Kaldi tdnn special case
        if (inputDesc[i].df == DF_NHWC && inputDesc[i].nDims == 3) {
            inputDesc[i] = tensor4df(inputDesc[i].dt, DF_NHWC, inputDesc[i].dims[2],
                inputDesc[i].dims[0], inputDesc[i].dims[1], 1);
        }
        int channel = inputDesc[i].nDims - 2;
        if (channel <= 0 || inputDesc[i].dims[channel] % 8 != 0) {
            continue;
        }
        U32 hw = 1;
        for (int j = 0; j < channel; j++) {
            hw *= inputDesc[i].dims[j];
        }
        if (hw != 1) {
            TensorDesc tmpDesc = outputDesc;
            if (tensorNumElements(inputDesc[i]) < tensorNumElements(outputDesc)) {
                tmpDesc = inputDesc[i];
                tmpDesc.df = outputDesc.df;
            }
            if (input.size() > 0) {
                CHECK_REQUIREMENT(tmpBytes >= tensorNumBytes(tmpDesc));
                tmpBytes -= tensorNumBytes(tmpDesc);
                CHECK_STATUS(transformFormat(inputDesc[i], input[i], tmpDesc, ptr));
                input[i] = ptr;
                ptr += tensorNumBytes(tmpDesc);
            }
            inputDesc[i] = tmpDesc;
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
    if (num == 2 && tensorNumElements(inputDesc[0]) != tensorNumElements(inputDesc[1])) {
        int nchwc8Count = 0;
        int nchwc16Count = 0;
        int id = -1;
        int align = 1;
        for (U32 i = 0; i < inputDesc.size(); i++) {
            if (inputDesc[i].df == DF_NCHWC8) {
                nchwc8Count++;
                id = i;
                align = 8;
            }
            if (inputDesc[i].df == DF_NCHWC16) {
                nchwc16Count++;
                id = i;
                align = 16;
            }
        }
        if ((nchwc8Count == 1 || nchwc16Count == 1) && (nchwc8Count + nchwc16Count == 1) &&
            !isSameDataFormat(outputDesc.df, DF_NCHW)) {
            for (int i = inputDesc[id].nDims; i > 0; i--) {
                inputDesc[id].dims[i] = inputDesc[id].dims[i - 1];
            }
            inputDesc[id].dims[0] = align;
            inputDesc[id].nDims++;
            inputDesc[id].dims[inputDesc[id].nDims - 2] /= align;
            inputDesc[id].df = inputDesc[1 - id].df;
            for (int i = inputDesc[1 - id].nDims; i > 0; i--) {
                inputDesc[1 - id].dims[i] = inputDesc[1 - id].dims[i - 1];
            }
            inputDesc[1 - id].dims[0] = 1;
            inputDesc[1 - id].nDims++;

            for (int i = outputDesc.nDims; i > 0; i--) {
                outputDesc.dims[i] = outputDesc.dims[i - 1];
            }
            outputDesc.dims[0] = align;
            outputDesc.nDims++;
            outputDesc.dims[outputDesc.nDims - 2] /= align;
        }
    }
#ifdef _DETAIL
    UNI_DETAIL_LOG("    data desc after align:\n");
    for (U32 i = 0; i < num; i++) {
        std::string line = "        input desc:" + tensorDesc2Str(inputDesc[i]);
        if (input.size() > 0) {
            U8 *p = (U8 *)input[i];
            float value;
            line += " data:";
            for (U32 j = 0; j < UNI_MIN(8, tensorNumElements(inputDesc[i])); j++) {
                transformToFloat(inputDesc[i].dt, p + bytesOf(inputDesc[i].dt) * j, &value, 1, 1);
                line += std::to_string(value) + " ";
            }
        }
        UNI_DETAIL_LOG("%s\n", line.c_str());
    }
    UNI_DETAIL_LOG("        output desc:%s\n", tensorDesc2Str(outputDesc).c_str());
#endif
    return ptr;
}

inline int useScalePower(TensorDesc a, TensorDesc b)
{
    std::vector<TensorDesc> descs = {a, b};
    std::vector<void *> inputs;
    TensorDesc tmp = a, scaleDesc;
    U32 tmpBytes = 0;
    align_param(descs, inputs, tmpBytes, nullptr, tmp);
    int id = -1;
    scale_axis(descs, tmp, &id, &scaleDesc);
    return id;
}
#endif
