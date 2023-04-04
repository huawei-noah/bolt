// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "error.h"
#include "tensor_desc.h"
#include "thread_affinity.h"

#define SIMDW 8
#define BYTES 4

inline U32 InferConvPointwiseUnrollOc(U32 oc)
{
    if ((oc % 24 != 0) && (oc % 32 == 0)) {
        return 32;
    } else {
        return 24;
    }
}

inline U32 InferConvDirectUnrollOc(U32 oc)
{
    if ((oc % 24 == 0) && (oc % 32 != 0)) {
        return 24;
    } else {
        return 32;
    }
}

inline I32 InferConvDirectOcBlockNum(I32 oc, I32 *ocbArray, I32 unrollOc, const I32 *unrollOcArray)
{
    I32 ocBlockNums = oc / unrollOc;
    I32 ocRemain = oc % unrollOc;
    I32 tmpSize = 0;
    for (I32 i = 0, j = 0; i < ocRemain; i += tmpSize, ++j) {
        tmpSize = unrollOcArray[((ocRemain - i) >> 3) - 1];
        ocbArray[j + 1] = tmpSize + ocbArray[j];
        ++ocBlockNums;
    }
    return ocBlockNums;
}

inline I32 InferConvDirectBolckIcDim(I32 originBlockIc, I32 unrollOc, I32 blockHwDim, I32 fh, I32 fw)
{
    if (fw * fh < 9) {
        return originBlockIc * 2;
    } else if (fw * fh > 9) {
        return originBlockIc / 2;
    }
    return originBlockIc;
}

inline U32 GetOcIdx(I32 ocBlockIdx, I32 oc, I32 unrollOc, I32 *ocbArray)
{
    I32 ocb = ocBlockIdx * unrollOc;
    if (ocBlockIdx > oc / unrollOc) {
        ocb = ocb + ocbArray[ocBlockIdx - oc / unrollOc] - (ocBlockIdx - oc / unrollOc) * unrollOc;
    }
    return ocb;
}

inline U32 InferConvBlockHW(U32 ohow, U32 originBlockHW, U32 alpha)
{
#ifdef _USE_OPENMP
    U32 beta = (ohow + alpha * originBlockHW - 1) / (alpha * originBlockHW);
    if (beta == 0) {
        return originBlockHW;
    }
    U32 blockHwDim = (ohow + alpha * beta - 1) / (alpha * beta);
#else
    U32 blockHwDim = originBlockHW;
#endif
    return blockHwDim;
}

inline EE InferConvWeightFormat(DataFormat &ftmDataFormat, U32 fnBlock)
{
    switch (fnBlock) {
        case 24: {
            ftmDataFormat = DF_NCHWCxN24;
            break;
        }
        case 32: {
            ftmDataFormat = DF_NCHWCxN32;
            break;
        }
        default:
            return NOT_MATCH;
    }
    return SUCCESS;
}

inline U32 CeilDivide(U32 a, U32 b)
{
    return (a + b - 1) / b;
}

inline EE tensor4dGetI32(
    TensorDesc desc, DataType *dt, DataFormat *df, I32 *num, I32 *numChannels, I32 *height, I32 *width)
{
    if (nullptr == num || nullptr == numChannels || nullptr == height || nullptr == width ||
        nullptr == dt || nullptr == df) {
        return NULL_POINTER;
    }
    if (4 != desc.nDims) {
        return NOT_MATCH;
    }

    *dt = desc.dt;
    *df = desc.df;
    *width = desc.dims[0];
    *height = desc.dims[1];
    *numChannels = desc.dims[2];
    *num = desc.dims[3];
    return SUCCESS;
}

inline I32 GetNewKernelDilatedPad(I32 iLength, I32 iPos, I32 kDilated, I32 dilate)
{
    I32 k = kDilated;
    if (iPos < 0) {
        k = kDilated + iPos;
        if (k > iLength) {
            k = (iLength - k % dilate) / dilate * dilate + k % dilate;
        }
    } else if (iPos + kDilated >= iLength) {
        k = UNI_MAX(iLength - iPos, 0);
    }
    return k;
}

inline I32 GetKernelnoDilated(I32 kDilated, I32 dilate)
{
    if (kDilated == 0 || kDilated == 1 || dilate == 1) {
        return kDilated;
    }
    return (kDilated - 1) / dilate + 1;
}

inline I32 JumpToWeightPos(I32 iPos, I32 dilate)
{
    if (iPos < 0) {
        return (-iPos + dilate - 1) / dilate;
    }
    return 0;
}

inline I32 JumpToInputPos(I32 iLength, I32 iPos, I32 kDilated, I32 dilate)
{
    if (iPos < 0) {
        return (iPos + kDilated - 1) % dilate;
    } else if (iPos + kDilated >= iLength && iLength > iPos) {
        return (iLength - iPos - 1) % dilate;
    }
    return 0;
}

template <typename T>
T gcd(T u, T v)
{
    while (v != 0) {
        T r = u % v;
        u = v;
        v = r;
    }
    return u;
}

#ifdef _USE_OPENMP
struct OpenMPController {
    bool useOmp;
    void checkAndSetOpenMP(I32 ohow, I32 threshold, I32 blockNums, I32 computeBlock, I32 thresholdB)
    {
#ifdef _WIN32
        if ((ohow < 8) && (blockNums < (OMP_NUM_THREADS * 8))) {
            useOmp = false;
        }
        if ((ohow < threshold) && (blockNums < (OMP_NUM_THREADS * 2))) {
            useOmp = false;
        }
        if ((OMP_NUM_THREADS > 4) && (computeBlock < thresholdB)) {
            useOmp = false;
        }
#endif
    }
    OpenMPController(): useOmp(true) {}
};
#endif