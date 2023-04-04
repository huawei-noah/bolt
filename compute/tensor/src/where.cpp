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
    int cloop = 1;
    for (U32 i = 1; i < conditionDesc.nDims; i++) {
        cloop *= conditionDesc.dims[i];
    }

    if (4 == xDesc.nDims && tensorNumElements(xDesc) == 1 && (tensorNumElements(outDesc) != tensorNumElements(conditionDesc))) {
        if (outDesc.dims[2] != conditionDesc.dims[2] || outDesc.dims[1] != conditionDesc.dims[1]) {
            for (U32 mov_n = 0; mov_n < outDesc.dims[3]; mov_n++) {
                for (U32 mov_c = 0; mov_c < outDesc.dims[2]; mov_c++) {
                     for (U32 mov_h = 0; mov_h < outDesc.dims[1]; mov_h++) {
                         for (U32 mov_w = 0; mov_w < outDesc.dims[0]; mov_w++) {
                             U32 y_index = mov_n * (outDesc.dims[2] * outDesc.dims[1] * outDesc.dims[0]) + mov_c * (outDesc.dims[1] * outDesc.dims[0]) + mov_h * outDesc.dims[0] + mov_w;
                             U32 con_index = mov_n * (conditionDesc.dims[2] * conditionDesc.dims[1] * conditionDesc.dims[0]);
                             if (outDesc.dims[2] == conditionDesc.dims[2]) {
                                 con_index += mov_c * (conditionDesc.dims[1] * conditionDesc.dims[0]);
                             } else {
                                 con_index += 0 * (conditionDesc.dims[1] * conditionDesc.dims[0]);
                             }

                             if (outDesc.dims[1] == conditionDesc.dims[1]) {
                                 con_index += mov_h * conditionDesc.dims[0];
                             } else {
                                 con_index += 0 * conditionDesc.dims[0];
                             }
                             con_index += mov_w;
                             out[y_index] = condition[con_index] ? x[0] : y[y_index];
                         }
                     }
                }
            }
        }
        return;
    }

    if ((tensorNumElements(xDesc) == 1) && 
        (tensorNumElements(yDesc) == tensorNumElements(outDesc)))
    {
        DataType odt, conDt;
        DataFormat odf, conDf;
        U32 on, oc, oh, ow;
        U32 conN, conC, conH, conW;
        if (tensorIs1d(conditionDesc)) {
            CHECK_STATUS(tensor1dGet(conditionDesc, &conDt, &conDf, &conN));
            conC = conH = conW = 1;
        } else if (tensorIs2d(conditionDesc)) {
            CHECK_STATUS(tensor2dGet(conditionDesc, &conDt, &conDf, &conN, &conC));
            conH = conW = 1;
        } else if (tensorIs3d(conditionDesc)) {
            CHECK_STATUS(tensor3dGet(conditionDesc, &conDt, &conDf, &conN, &conC, &conW));
            conH = 1;
        } else if (tensorIs4d(conditionDesc)) {
            CHECK_STATUS(tensor4dGet(conditionDesc, &conDt, &conDf, &conN, &conC, &conH, &conW));
        } else {
            UNI_ERROR_LOG("where currently only support 1d/2d/3d/4d tensor.\n");
            return;
        }
        if (tensorIs1d(outDesc)) {
            CHECK_STATUS(tensor1dGet(outDesc, &odt, &odf, &on));
            oc = oh = ow = 1;
        } else if (tensorIs2d(outDesc)) {
            CHECK_STATUS(tensor2dGet(outDesc, &odt, &odf, &on, &oc));
            oh = ow = 1;
        } else if (tensorIs3d(outDesc)) {
            CHECK_STATUS(tensor3dGet(outDesc, &odt, &odf, &on, &oc, &ow));
            oh = 1;
        } else if (tensorIs4d(outDesc)) {
            CHECK_STATUS(tensor4dGet(outDesc, &odt, &odf, &on, &oc, &oh, &ow));
        } else {
            UNI_ERROR_LOG("where currently only support 1d/2d/3d/4d tensor.\n");
            return;
        }
        bool nflag = (conN == on);
        bool cflag = (conC == oc);
        bool hflag = (conH == oh);
        bool wflag = (conW == ow);
        U8 cx = 1;
        U8 conCx = 1;
        if (odf == DF_NCHWC8) {
            cx = 8;
        }
        if (odf == DF_NCHWC16) {
            cx = 16;
        }
        if (conDf == DF_NCHWC8) {
            conCx = 8;
        }
        if (conDf == DF_NCHWC16) {
            conCx = 16;
        }
        oc /= cx;
        conC /= conCx;
        if ((cx == 1) && (conCx == 1)) {
            for (U32 n = 0; n < on; n++) {
                U32 cn = n * nflag;
                for (U32 c = 0; c < oc; c++) {
                    U32 cc = c * cflag;
                    for (U32 h = 0; h < oh; h++) {
                        U32 ch = h * hflag;
                        if (wflag == 0) {
                            U32 idx = ((n * oc + c) * oh + h) * ow;;
                            if (condition[(cn * conC + cc) * conH + ch]) {
                                for (U32 w = 0; w < ow; ++w) {
                                    out[idx + w] = x[0];
                                }
                            } else {
                                UNI_MEMCPY(out + idx, y + idx, ow * sizeof(T));
                            }
                        } else {
                            for (U32 w = 0; w < ow; w++) {
                                U32 idx = ((n * oc + c) * oh + h) * ow + w;
                                if (condition[((cn * conC + cc) * conH + ch) * conW + w]) {
                                    out[idx] = x[0];
                                } else {
                                    out[idx] = y[idx];
                                }
                            }
                        }
                    }
                }
            }
        } else {
            for (U32 n = 0; n < on; n++) {
                U32 cn = n * nflag;
                for (U32 c = 0; c < oc; c++) {
                    for (U32 h = 0; h < oh; h++) {
                        U32 ch = h * hflag;
                        for (U32 w = 0; w < ow; w++) {
                            U32 cw = w * wflag;
                            for (U32 ci = 0; ci < cx; ++ci) {
                                U32 idx = (((n * oc + c) * oh + h) * ow + w) * cx + ci;
                                U32 conc = (c * cx + ci) / conCx * cflag;
                                U32 conci = (c * cx + ci) % conCx * cflag;
                                if (condition[(((cn * conC + conc) * conH + ch) * conW + cw) * conCx + conci]) {
                                    out[idx] = x[0];
                                } else {
                                    out[idx] = y[idx];
                                }
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

EE where_infer_output_size(
    Tensor *cTensor, Tensor *xTensor, Tensor *yTensor, Tensor *outputTensor, ArchInfo_t archInfo)
{
    TensorDesc cDesc = cTensor->get_desc();
    TensorDesc xDesc = xTensor->get_desc();
    TensorDesc yDesc = yTensor->get_desc();
    TensorDesc outDesc = (xDesc.nDims > yDesc.nDims) ? xDesc : yDesc;
    outDesc = (cDesc.nDims > outDesc.nDims) ? cDesc : outDesc;
    for (U32 i = 0; i < cDesc.nDims; i++) {
        int max_value = UNI_MAX(outDesc.dims[i], cDesc.dims[i]);
        int min_value = UNI_MIN(outDesc.dims[i], cDesc.dims[i]);
        if (min_value == 1) {
            outDesc.dims[i] = max_value;
        } else {
            outDesc.dims[i] = min_value;
        }
    }
    for (U32 i = 0; i < xDesc.nDims; i++) {
        int max_value = UNI_MAX(outDesc.dims[i], xDesc.dims[i]);
        int min_value = UNI_MIN(outDesc.dims[i], xDesc.dims[i]);
        if (min_value == 1) {
            outDesc.dims[i] = max_value;
        } else {
            outDesc.dims[i] = min_value;
        }
    }
    for (U32 i = 0; i < yDesc.nDims; i++) {
        int max_value = UNI_MAX(outDesc.dims[i], yDesc.dims[i]);
        int min_value = UNI_MIN(outDesc.dims[i], yDesc.dims[i]);
        if (min_value == 1) {
            outDesc.dims[i] = max_value;
        } else {
            outDesc.dims[i] = min_value;
        }
    }
#ifdef _USE_CPU
    if (IS_CPU(archInfo->arch) && tensorIsShape(cDesc) && tensorIsShape(xDesc) && tensorIsShape(yDesc) &&
        tensorIsShape(outDesc)) {
        where_kernel<U32>(cDesc, (const U8 *)(cDesc.dims + cDesc.nDims), xDesc,
            (const U32 *)(xDesc.dims + xDesc.nDims), yDesc, (const U32 *)(yDesc.dims + yDesc.nDims),
            outDesc, (U32 *)(outDesc.dims + outDesc.nDims));
    }
#endif
    outputTensor->resize(outDesc);
    return SUCCESS;
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
        case DT_I32:
        case DT_U32: {
            where_kernel<U32>(conditionDesc, (const U8 *)condition, xDesc, (const U32 *)x, yDesc,
                (const U32 *)y, outDesc, (U32 *)out);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
