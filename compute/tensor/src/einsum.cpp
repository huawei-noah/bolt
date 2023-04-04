// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <unordered_map>
#include "blas_enhance.h"
#include "tensor_computing.h"
#ifdef _USE_CPU
#include "cpu/tensor_computing_cpu.h"
#endif
#ifdef _USE_MALI
#include "gpu/mali/tensor_computing_mali.h"
#endif

inline U32 reduceDims(U32 *dims, U32 s, U32 e)
{
    U32 res = 1;
    for (U32 i = s; i < e; ++i) {
        res *= dims[i];
    }
    return res;
}

EE einsum(std::vector<Tensor> inTensors,
    EinsumParamSpec p,
    Tensor outputTensor,
    Tensor tmpTensor,
    ArchInfo_t archInfo)
{
    if (p.num_equation_o == 0 || inTensors.size() < 2) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    TensorDesc lDesc = inTensors[0].get_desc();
    F32 *lop = (F32 *)get_ptr_from_tensor(inTensors[0], archInfo->arch);
    TensorDesc rDesc = inTensors[1].get_desc();
    F32 *rop = (F32 *)get_ptr_from_tensor(inTensors[1], archInfo->arch);
    TensorDesc oDesc = outputTensor.get_desc();
    F32 *output = (F32 *)get_ptr_from_tensor(outputTensor, archInfo->arch);
    U32 tmpBytes = tmpTensor.bytes();
    void *tmp = get_ptr_from_tensor(tmpTensor, archInfo->arch);

    std::unordered_map<char, U32> outputChMap;
    std::unordered_map<char, U32> lMap;
    for (int i = 0; i < p.num_equation_o; ++i) {
        outputChMap[p.equation_o[i]] = i;
    }

    bool needTranspose = false;
    U32 lreduceDim = 0;
    U32 lidx = 0;
    U32 lastIdx = 0;
    for (int j = 0; j < p.num_equation_l; ++j) {
        if (!outputChMap.count(p.equation_l[j])) {
            lidx = j;
            lreduceDim = lDesc.dims[lidx];
        } else {
            if (outputChMap[p.equation_l[j]] < lastIdx) {
                needTranspose = true;
            }
            lMap[p.equation_l[j]] = j;
            lastIdx = outputChMap[p.equation_l[j]];
        }
    }
    U32 rreduceDim = 0;
    U32 ridx = 0;
    lastIdx = 0;
    int alignDim = -1;
    int lalignDim = -1;
    int ralignDim = -1;
    for (int j = 0; j < p.num_equation_r; ++j) {
        if (!outputChMap.count(p.equation_r[j])) {
            ridx = j;
            rreduceDim = rDesc.dims[ridx];
        } else {
            if (outputChMap[p.equation_r[j]] < lastIdx) {
                needTranspose = true;
            }
            lastIdx = outputChMap[p.equation_r[j]];
            if ((lMap.count(p.equation_r[j])) && (alignDim == -1)) {
                alignDim = outputChMap[p.equation_r[j]];
                lalignDim = lMap[p.equation_r[j]];
                ralignDim = j;
            }
        }
    }

    if (lreduceDim != rreduceDim) {
        UNI_ERROR_LOG("Einsum currently not support (%s,%s->%s).\n", p.equation_l, p.equation_r,
            p.equation_o);
        return NOT_SUPPORTED;
    }

    if (int(ridx) > ralignDim || int(lidx) > lalignDim) {
        needTranspose = true;
    }

    if (needTranspose) {
        U32 ldims[8];
        U32 rdims[8];
        for (U32 i = 0; i < tensorNumElements(oDesc); ++i) {
            std::vector<U32> idx = calculateLocalIndex(i, oDesc.dims, oDesc.nDims);
            for (int j = 0; j < p.num_equation_l; ++j) {
                if (outputChMap.count(p.equation_l[j])) {
                    ldims[j] = idx[outputChMap[p.equation_l[j]]];
                }
            }
            for (int j = 0; j < p.num_equation_r; ++j) {
                if (outputChMap.count(p.equation_r[j])) {
                    rdims[j] = idx[outputChMap[p.equation_r[j]]];
                }
            }
            F32 res = 0;
            for (U32 al = 0; al < lreduceDim; ++al) {
                ldims[lidx] = al;
                rdims[ridx] = al;
                U32 lopIdx = calculateGlobalIndex(ldims, lDesc.dims, lDesc.nDims);
                U32 ropIdx = calculateGlobalIndex(rdims, rDesc.dims, rDesc.nDims);
                res += lop[lopIdx] * rop[ropIdx];
            }
            output[i] = res;
        }
    } else {
        TensorDesc mmmrDesc;
        U32 M, N;
        U32 K = rDesc.dims[ridx];
        if (ridx != 0) {
            ralignDim = ridx + 1;
            alignDim = UNI_MIN(alignDim, int(outputChMap[p.equation_r[ralignDim]]));
            N = reduceDims(rDesc.dims, 0, ridx);
            mmmrDesc = tensor2df(rDesc.dt, DF_NORMAL, K, N);
        } else {
            N = reduceDims(rDesc.dims, 1, ralignDim);
            mmmrDesc = tensor2df(rDesc.dt, DF_TRANSPOSE, N, K);
        }
        TensorDesc mmmlDesc;
        if (lidx != 0) {
            lalignDim = lidx + 1;
            alignDim = UNI_MIN(alignDim, int(outputChMap[p.equation_l[lalignDim]]));
            M = reduceDims(lDesc.dims, 0, lidx);
            mmmlDesc = tensor2df(lDesc.dt, DF_TRANSPOSE, K, M);
        } else {
            M = reduceDims(lDesc.dims, 1, lalignDim);
            mmmlDesc = tensor2df(lDesc.dt, DF_NORMAL, M, K);
        }
        TensorDesc mmmODesc = tensor2df(oDesc.dt, DF_NORMAL, M, N);

        U32 outputTilesNum = reduceDims(oDesc.dims, alignDim, oDesc.nDims);
        U32 outputTile = reduceDims(oDesc.dims, 0, alignDim);
        U32 lTilesNum = reduceDims(lDesc.dims, lalignDim, lDesc.nDims);
        U32 lTile = reduceDims(lDesc.dims, 0, lalignDim);
        U32 rTilesNum = reduceDims(rDesc.dims, ralignDim, rDesc.nDims);
        U32 rTile = reduceDims(rDesc.dims, 0, ralignDim);

        UNI_MEMSET(output, 0, tensorNumBytes(oDesc));
        if (lTilesNum == rTilesNum) {
            for (U32 i = 0; i < outputTilesNum; ++i) {
                F32 *curOutput = output + i * outputTile;
                F32 *curLop = lop + (i % lTilesNum) * lTile;  // special case
                F32 *curRop = rop + (i % rTilesNum) * rTile;  // special case
                CHECK_STATUS(matrix_matrix_multiply(mmmlDesc, curLop, mmmrDesc, curRop, tmpBytes,
                    tmp, mmmODesc, curOutput, nullptr, archInfo->arch));
            }
        } else {
            U32 ldims[8];
            U32 rdims[8];
            U32 odims[8];
            std::unordered_map<char, U32> laMap, raMap;
            for (int i = 0; i < p.num_equation_l; ++i) {
                laMap[p.equation_l[i]] = i;
            }
            for (int i = 0; i < p.num_equation_r; ++i) {
                raMap[p.equation_r[i]] = i;
            }
            U32 alignLen = oDesc.nDims - alignDim;
            for (U32 i = 0; i < alignLen; ++i) {
                odims[i] = oDesc.dims[i + alignDim];
                if (laMap.count(p.equation_o[i])) {
                    ldims[i] = oDesc.dims[i];
                } else {
                    ldims[i] = 1;
                }
                if (raMap.count(p.equation_o[i])) {
                    rdims[i] = oDesc.dims[i];
                } else {
                    rdims[i] = 1;
                }
            }

            for (U32 i = 0; i < outputTilesNum; ++i) {
                F32 *curOutput = output + i * outputTile;
                U32 lidx = 0;
                U32 ridx = 0;
                U32 oidx = i;
                U32 ldim = 1;
                U32 rdim = 1;
                for (U32 j = 0; j < alignLen; ++j) {
                    U32 dimIdx = oidx % odims[j];
                    oidx /= odims[j];
                    lidx += dimIdx % ldims[j] * ldim;
                    ridx += dimIdx % rdims[j] * rdim;
                    ldim *= ldims[j];
                    rdim *= rdims[j];
                }
                F32 *curLop = lop + lidx * lTile;
                F32 *curRop = rop + ridx * rTile;
                matrix_matrix_multiply(mmmlDesc, curLop, mmmrDesc, curRop, tmpBytes, tmp, mmmODesc,
                    curOutput, nullptr, archInfo->arch);
            }
        }
    }

    return SUCCESS;
}

EE einsum_infer_output_size(
    std::vector<Tensor *> inTensors, EinsumParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo)
{
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }

    if (p.num_equation_o == 0 && p.num_equation_l == 1 && p.num_equation_r == 0) {
        outputTensor->resize(inTensors[0]->get_desc());
        return SUCCESS;
    }

    std::unordered_map<char, int> l_map, r_map;
    TensorDesc lDesc = inTensors[0]->get_desc();
    TensorDesc oDesc = lDesc;
    oDesc.nDims = p.num_equation_o;

    for (int j = 0; j < p.num_equation_l; ++j) {
        l_map[p.equation_l[j]] = lDesc.dims[j];
    }
    if (inTensors.size() > 1) {
        TensorDesc rDesc = inTensors[1]->get_desc();
        for (int j = 0; j < p.num_equation_r; ++j) {
            r_map[p.equation_r[j]] = rDesc.dims[j];
        }
    }

    for (int i = 0; i < p.num_equation_o; ++i) {
        oDesc.dims[i] = l_map.count(p.equation_o[i]) ? l_map[p.equation_o[i]]
                                                     : r_map[p.equation_o[i]];
    }

    if (oDesc.nDims > 3) {
        oDesc.df = DF_NCHW;
    } else if (oDesc.nDims == 3) {
        oDesc.df = DF_MTK;
    } else {
        oDesc.df = DF_NORMAL;
    }

    outputTensor->resize(oDesc);
    return SUCCESS;
}

EE einsum_infer_forward_tmp_bytes(
    std::vector<Tensor> inTensors, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo)
{
    TensorDesc lDesc = inTensors[0].get_desc();
    TensorDesc rDesc = inTensors[1].get_desc();
    *bytes = tensorNumBytes(lDesc) + tensorNumBytes(lDesc);
    return SUCCESS;
}
