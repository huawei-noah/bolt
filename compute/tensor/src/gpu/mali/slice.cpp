// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "sys.h"
#include "types.h"
#include "tensor_desc.h"
#include "error.h"
#include "gpu/mali/tensor_computing_mali.h"
#include "gpu/mali/fp16/slice_mali_fp16.h"

EE slice_infer_output_size_mali(TensorDesc inputDesc,
    SliceParamSpec p,
    std::vector<TensorDesc> *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    if (outputDesc == NULL) {
        CHECK_STATUS(NULL_POINTER);
    }
    int axis = p.axis;
    int *slice_points = p.slice_points;
    U32 num = outputDesc->size();
    axis = (axis + inputDesc.nDims) % inputDesc.nDims;
    I32 target_axis = inputDesc.nDims - 1 - axis;
    for (U32 i = 0; i < num; i++) {
        (*outputDesc)[i] = inputDesc;

        I32 prev_point = 0;
        if (i > 0) {
            prev_point = slice_points[i - 1];
        }
        I32 next_point = inputDesc.dims[target_axis];
        if (i < num - 1) {
            next_point = slice_points[i];
        }
        if (prev_point < 0) {
            prev_point = (prev_point + inputDesc.dims[target_axis]) % inputDesc.dims[target_axis];
        }
        if (next_point < 0) {
            next_point = (next_point + inputDesc.dims[target_axis]) % inputDesc.dims[target_axis];
        }
        (*outputDesc)[i].dims[target_axis] = next_point - prev_point;
    }
    if (inputDesc.df == DF_MKT) {
        if (axis == 2) {  // slice on T
            DataType dt;
            U32 m, k, t;
            U32 gw, gh, gc;
            get_nlp_mkt_val(inputDesc, &dt, &m, &k, &t);
            map_nlp_mkt_to_ncwhc4(m, k, t, &gw, &gh, &gc);
            CHECK_STATUS(infer_gclmem_desc_ncwhc4(
                gw, gh, gc * 4, 0, 0, 0, 0, 0, dt, dt, gclmemInputDesc, NULL));
            if (gclmemOutputDesc) {
                for (U32 i = 0; i < num; ++i) {
                    get_nlp_mkt_val((*outputDesc)[i], NULL, &m, &k, &t);
                    map_nlp_mkt_to_ncwhc4(m, k, t, &gw, &gh, &gc);
                    CHECK_STATUS(infer_gclmem_desc_ncwhc4(
                        0, 0, 0, 0, 0, gw, gh, gc * 4, dt, dt, NULL, &gclmemOutputDesc[i]));
                }
            }
        }
        return SUCCESS;
    }
    return NOT_SUPPORTED;
}

inline EE slice_checkpara_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    SliceParamSpec p,
    std::vector<TensorDesc> outputDesc,
    std::vector<void *> *output)
{
    if (handle == nullptr || input == nullptr) {
        return NULL_POINTER;
    }
    if (input->desc.memFormat != DF_NCWHC4) {
        return NOT_SUPPORTED;
    }
    for (auto p : (*output)) {
        if (p == nullptr) {
            return NULL_POINTER;
        }
        if (((GCLMem_t)p)->desc.memFormat != input->desc.memFormat) {
            return NOT_MATCH;
        }
    }
    if (inputDesc.df != DF_MKT) {
        return NOT_SUPPORTED;
    }
    if (inputDesc.df == DF_MKT && p.axis != 2) {
        return NOT_SUPPORTED;
    }
    for (auto p : outputDesc) {
        if (p.df != inputDesc.df) {
            return NOT_MATCH;
        }
    }
    return SUCCESS;
}

EE slice_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    SliceParamSpec p,
    std::vector<TensorDesc> outputDesc,
    std::vector<void *> *output)
{
    EE ret = SUCCESS;
    CHECK_STATUS(slice_checkpara_mali(handle, inputDesc, input, p, outputDesc, output));
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = slice_mali_fp16(handle, inputDesc, input, p, outputDesc, output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
