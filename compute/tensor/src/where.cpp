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

EE where_infer_output_size(
    Tensor *xTensor, Tensor *yTensor, Tensor *outputTensor, ArchInfo_t archInfo)
{
    TensorDesc xDesc = xTensor->get_desc();
    TensorDesc yDesc = yTensor->get_desc();
    TensorDesc outDesc = (xDesc.nDims > yDesc.nDims) ? xDesc : yDesc;
    for (U32 i = 0; i < xDesc.nDims; i++) {
        if (xDesc.dims[i] > outDesc.dims[i]) {
            outDesc.dims[i] = xDesc.dims[i];
        }
    }
    for (U32 i = 0; i < yDesc.nDims; i++) {
        if (yDesc.dims[i] > outDesc.dims[i]) {
            outDesc.dims[i] = yDesc.dims[i];
        }
    }
    outputTensor->resize(outDesc);
    return SUCCESS;
}

inline static std::vector<U32> get_dims(const TensorDesc &desc)
{
    std::vector<U32> dims;
    if (desc.df == DF_NCHWC8) {
        dims.push_back(8);
    }
    for (U32 i = 0; i < desc.nDims; i++) {
        dims.push_back(desc.dims[i]);
    }
    return dims;
}

template <typename T>
static void where_kernel(const TensorDesc &conditionDesc,
    const U8 *condition,
    const TensorDesc &xDesc,
    const T *x,
    const TensorDesc &yDesc,
    const T *y,
    const TensorDesc &outDesc,
    T *out)
{
    if (tensorNumElements(xDesc) == 1 &&
        tensorNumElements(conditionDesc) >= outDesc.dims[0] &&
        tensorNumElements(yDesc) == tensorNumElements(outDesc))
    {
        UNI_MEMCPY(out, y, tensorNumBytes(yDesc));
        DataType odt;
        DataFormat odf;
        U32 on, oc, oh, ow;
        if (tensorIs3d(outDesc)) {
            CHECK_STATUS(tensor3dGet(outDesc, &odt, &odf, &on, &oc, &ow));
            oh = 1;
        } else if (tensorIs4d(outDesc)) {
            CHECK_STATUS(tensor4dGet(outDesc, &odt, &odf, &on, &oc, &oh, &ow));
        } else {
            UNI_ERROR_LOG("where currently only support 3d/4d tensor.\n");
            return;
        }
        U8 c8 = 1;
        if (odf == DF_NCHWC8) {
            c8 = 8;
        }
        oc /= c8;
        for (U32 w = 0; w < ow; w++) {
            if (condition[w]) {
                for (U32 n = 0; n < on; n++) {
                    for (U32 c0 = 0; c0 < oc; c0++) {
                        for (U32 h = 0; h < oh; h++) {
                            for (U32 c1 = 0; c1 < c8; c1++) {
                                out[(((n * oc + c0) * oh + h) * ow + w) * c8 + c1] = x[0];
                            }
                        }
                    }
                }
            }
        }
        return;
    }
    U32 length = tensorNumElements(outDesc);
    if (xDesc.df != DF_NCHWC8 && yDesc.df != DF_NCHWC8) {
        for (U32 i = 0; i < length; i++) {
            const std::vector<U32> &id = calculateLocalIndex(i, outDesc.dims, outDesc.nDims);
            int ci = calculateGlobalIndex(id.data(), conditionDesc.dims, conditionDesc.nDims);
            int xi = calculateGlobalIndex(id.data(), xDesc.dims, xDesc.nDims);
            int yi = calculateGlobalIndex(id.data(), yDesc.dims, yDesc.nDims);
            out[i] = condition[ci] ? x[xi] : y[yi];
        }
        return;
    }
    const std::vector<U32> &cdims = get_dims(conditionDesc);
    const std::vector<U32> &xdims = get_dims(xDesc);
    const std::vector<U32> &ydims = get_dims(yDesc);
    const std::vector<U32> &odims = get_dims(outDesc);
    std::vector<U32> id_c1(odims.size()), id_c8(odims.size() + 1);
    U32 *cid = (conditionDesc.nDims == cdims.size()) ? id_c1.data() : id_c8.data();
    U32 *xid = (xDesc.nDims == xdims.size()) ? id_c1.data() : id_c8.data();
    U32 *yid = (yDesc.nDims == ydims.size()) ? id_c1.data() : id_c8.data();
    int axis = outDesc.nDims - 2;
    for (U32 i = 0; i < length; i++) {
        const std::vector<U32> &id = calculateLocalIndex(i, odims.data(), odims.size());
        if (outDesc.nDims != odims.size()) {
            UNI_MEMCPY(id_c8.data(), id.data(), id.size() * sizeof(float));
            UNI_MEMCPY(id_c1.data(), id.data() + 1, (id.size() - 1) * sizeof(float));
            id_c1[axis] = id_c1[axis] * 8 + id[0];
        } else {
            UNI_MEMCPY(id_c1.data(), id.data(), id.size() * sizeof(float));
            UNI_MEMCPY(id_c8.data() + 1, id.data(), id.size() * sizeof(float));
            id_c8[0] = id[axis] % 8;
            id_c8[axis + 1] = id[axis] / 8;
        }
        int ci = calculateGlobalIndex(cid, cdims.data(), cdims.size());
        int xi = calculateGlobalIndex(xid, xdims.data(), xdims.size());
        int yi = calculateGlobalIndex(yid, ydims.data(), ydims.size());
        out[i] = condition[ci] ? x[xi] : y[yi];
    }
}

EE where(
    Tensor conditionTensor, Tensor xTensor, Tensor yTensor, Tensor outputTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    void *condition = get_ptr_from_tensor(conditionTensor, arch);
    void *x = get_ptr_from_tensor(xTensor, arch);
    void *y = get_ptr_from_tensor(yTensor, arch);
    void *out = get_ptr_from_tensor(outputTensor, arch);
    TensorDesc conditionDesc = conditionTensor.get_desc();
    TensorDesc xDesc = xTensor.get_desc();
    TensorDesc yDesc = yTensor.get_desc();
    TensorDesc outDesc = outputTensor.get_desc();

    EE ret = SUCCESS;
    switch (xDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            where_kernel<F32>(conditionDesc, (const U8 *)condition, xDesc, (const F32 *)x, yDesc,
                (const F32 *)y, outDesc, (F32 *)out);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            where_kernel<F16>(conditionDesc, (const U8 *)condition, xDesc, (const F16 *)x, yDesc,
                (const F16 *)y, outDesc, (F16 *)out);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
