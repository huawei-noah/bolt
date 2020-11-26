// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TENSOR_DESC
#define _H_TENSOR_DESC
#include <limits.h>
#include <vector>
#include <string>
#include <string.h>
#include "error.h"

#define UNUSED(x) (void)x
#define UNI_MIN(a, b) (((a) < (b)) ? (a) : (b))
#define UNI_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define UNI_ABS(a) (((a) > 0) ? (a) : (-1 * (a)))
#define UNI_F16_MIN -65504.0f
#define UNI_F16_MAX 65504.0f
#define NAME_LEN 128
#ifdef __cplusplus
extern "C" {
#endif
int UNI_ISNAN(float a);
int UNI_ISINF(float a);
#ifdef __cplusplus
}
#endif

#ifdef _USE_X86
#include <immintrin.h>
#endif
#if defined(_USE_NEON) || defined(_USE_MALI)
#include <arm_neon.h>
#ifdef __aarch64__
typedef __fp16 F16;
#endif
typedef int8_t INT8;
#else
typedef char INT8;
#endif
typedef unsigned char U8;
typedef const unsigned char CU8;
typedef char I8;
typedef const char CI8;
typedef unsigned int U32;
typedef const unsigned int CU32;
typedef int I32;
typedef const int CI32;
typedef float F32;
typedef double F64;
typedef long I64;
typedef unsigned char BIN8;

typedef enum {
    RGB_SC = 0,  // scale and center crop
    RGB = 1,
    BGR = 2,
    RGB_RAW = 3,
    RGB_SC_RAW = 4,
    BGR_SC_RAW = 5
} ImageFormat;

typedef enum {
    DT_U8 = 0,
    DT_I8 = 1,
    DT_U32 = 2,
    DT_I32 = 3,
    DT_F16 = 4,
    DT_F16_8Q = 5,
    DT_F32 = 6,
    DT_BIN01 = 7,
    DT_BIN11 = 8,
    DT_NUM = 9
} DataType;

inline U32 bytesOf(DataType dt)
{
    U32 bytes[] = {1, 1, 4, 4, 2, 2, 4, 1, 1,
        8};  // Please divide number of elements by 8 first in the case of binary data types
    return dt < DT_NUM ? bytes[dt] : 0;
}

typedef enum {
    DF_NCHW,
    DF_NCHWN16,     // vectorize for N=16, for filter
    DF_NCHWC8,      // vectorize for C=8, for input and output
    DF_HWNCN16,     // vectorize for N=16, for filter in winograd
    DF_NHWCN16,     // im2col + GEMM, for filter
    DF_NHWCN8,      // vectorize for N=8, not used
    DF_HWNCN8C4,    // int8 filter for winograd
    DF_NCHWN8C4,    // int8 im2col + GEMM, for filter
    DF_NCHWN8HW4,   // int8 im2col + GEMM in the first layer, for filter
    DF_NCHWN16C8,   // bnn im2col + GEMM, for filter
    DF_NCHWCxN32,   // x86 AVX2 direct conv, for filter
    DF_NCHWCxN24,   // x86 AVX2 conv 1x1, for filter
    DF_NCHWC24,     // x86 AVX2 depthwise conv, for filter
    DF_TRANSPOSE,   // vectorize for COL_MAJOR
    DF_NORMAL,      // vectorize for ROW_MAJOR
    DF_MTK,         // RNN input, M: batch, T: step, K: x_dim
    DF_MKT,         // RNN input, M: batch, T: step, K: x_dim
    DF_NK,          // MMM/MVM filter, N: col_num, K: row_num
    DF_NKN16,       // MMM/MVM filter, vectorized for N=16
    DF_NKN32,       // MMM/MVM filter, vectorized for N=32
    DF_NKN64,       // MMM/MVM filter, vectorized for N=64
    DF_NKN32K4,     // int8 MVM filter, vectorized for N=32
    DF_NCWHC4,      // ocl mali input and output
    DF_NCHWC3,      // ocl mali support input rgb
    DF_NHWC,        // ocl mali support input/output
    DF_NCHWN4C4,    // ocl mali conv filter
    DF_NCHWN4,      // ocl mali conv filter
    DF_HWCN,        // ocl mali filter
    DF_NCWHN4C4,    // ocl mali fc   filter
    DF_NHWCN4,      // ocl mali filter
    DF_CHWNC4,      // ocl mali filter
    DF_CHWNC8,      // ocl mali filter
    DF_CHWNC16,     // ocl mali filter
    DF_CHWC8_NCN8,  // fp32 dw_conv, vectorized for C8 and N8
    DF_RGB,
    DF_HWNCN8,  // fp32 filter for winograd
    DF_NKN24,   // Optimized MMM filter for FP16
#ifdef __aarch64__
    DF_NKN12,  // Optimized MMM filter for FP32
#else
    DF_NKN8,  // Optimized MMM filter for FP32
#endif
    DF_NKN12K4,  // Optimized MMM filter for INT8
    DF_NCTHW,    // conv 3d
} DataFormat;

typedef struct {
    DataType dt = DT_U8;
    DataFormat df;
    U32 nDims = 0;
    U32 dims[6] = {0};
} TensorDesc;

inline TensorDesc tensor5df(
    DataType dt, DataFormat df, U32 num, U32 numChannels, U32 time, U32 height, U32 width)
{
    TensorDesc ret;
    ret.dt = dt;
    ret.df = df;
    ret.nDims = 5;
    ret.dims[0] = width;
    ret.dims[1] = height;
    ret.dims[2] = time;
    ret.dims[3] = numChannels;
    ret.dims[4] = num;
    return ret;
}

inline TensorDesc tensor5d(DataType dt, U32 num, U32 numChannels, U32 time, U32 height, U32 width)
{
    return tensor5df(dt, DF_NCHW, num, numChannels, time, height, width);
}

inline TensorDesc tensor4df(
    DataType dt, DataFormat df, U32 num, U32 numChannels, U32 height, U32 width)
{
    TensorDesc ret;
    ret.dt = dt;
    ret.df = df;
    ret.nDims = 4;
    ret.dims[0] = width;
    ret.dims[1] = height;
    ret.dims[2] = numChannels;
    ret.dims[3] = num;
    return ret;
}

inline TensorDesc tensor4d(DataType dt, U32 num, U32 numChannels, U32 height, U32 width)
{
    return tensor4df(dt, DF_NCHW, num, numChannels, height, width);
}

inline TensorDesc tensor3df(DataType dt, DataFormat df, U32 numChannels, U32 height, U32 width)
{
    TensorDesc ret = tensor4df(dt, df, 1, numChannels, height, width);
    ret.nDims = 3;
    return ret;
}

inline TensorDesc tensor3d(DataType dt, U32 numChannels, U32 height, U32 width)
{
    return tensor3df(dt, DF_NCHW, numChannels, height, width);
}

inline TensorDesc tensor2df(DataType dt, DataFormat df, U32 numRows, U32 numColumns)
{
    TensorDesc ret = tensor3df(dt, df, 1, numRows, numColumns);
    ret.nDims = 2;
    return ret;
}

inline TensorDesc tensor2d(DataType dt, U32 numRows, U32 numColumns)
{
    TensorDesc ret = tensor3d(dt, 1, numRows, numColumns);
    ret.nDims = 2;
    return ret;
}

inline TensorDesc tensor1d(DataType dt, U32 len)
{
    TensorDesc ret = tensor2d(dt, 1, len);
    ret.nDims = 1;
    return ret;
}

inline EE tensor1dGet(TensorDesc desc, DataType *dt, DataFormat *df, U32 *len)
{
    if (nullptr == len || nullptr == dt || nullptr == df) {
        return NULL_POINTER;
    }
    if (1 != desc.nDims) {
        return NOT_MATCH;
    }

    *df = desc.df;
    *dt = desc.dt;
    *len = desc.dims[0];
    return SUCCESS;
}

inline EE tensor2dGet(TensorDesc desc, DataType *dt, DataFormat *df, U32 *numRows, U32 *numColumns)
{
    if (nullptr == numColumns || nullptr == numRows || nullptr == dt || nullptr == df) {
        return NULL_POINTER;
    }
    if (2 != desc.nDims) {
        return NOT_MATCH;
    }

    *df = desc.df;
    *dt = desc.dt;
    *numColumns = desc.dims[0];
    *numRows = desc.dims[1];
    return SUCCESS;
}

inline EE tensor3dGet(
    TensorDesc desc, DataType *dt, DataFormat *df, U32 *numChannels, U32 *height, U32 *width)
{
    if (nullptr == numChannels || nullptr == height || nullptr == width || nullptr == dt ||
        nullptr == df) {
        return NULL_POINTER;
    }
    if (3 != desc.nDims) {
        return NOT_MATCH;
    }

    *dt = desc.dt;
    *df = desc.df;
    *width = desc.dims[0];
    *height = desc.dims[1];
    *numChannels = desc.dims[2];
    return SUCCESS;
}

inline EE tensor4dGet(
    TensorDesc desc, DataType *dt, DataFormat *df, U32 *num, U32 *numChannels, U32 *height, U32 *width)
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

inline EE tensor5dGet(TensorDesc desc,
    DataType *dt,
    DataFormat *df,
    U32 *num,
    U32 *numChannels,
    U32 *time,
    U32 *height,
    U32 *width)
{
    if (nullptr == num || nullptr == numChannels || nullptr == time || nullptr == height ||
        nullptr == width || nullptr == dt || nullptr == df) {
        return NULL_POINTER;
    }
    if (5 != desc.nDims) {
        return NOT_MATCH;
    }

    *dt = desc.dt;
    *df = desc.df;
    *width = desc.dims[0];
    *height = desc.dims[1];
    *time = desc.dims[2];
    *numChannels = desc.dims[3];
    *num = desc.dims[4];
    return SUCCESS;
}

inline EE tensorSelectGet(TensorDesc desc,
    DataType *dt,
    DataFormat *df,
    U32 *num,
    U32 *numChannels,
    U32 *height,
    U32 *width,
    U32 *time = NULL)

{
    U32 ndims = desc.nDims;
    if (dt) {
        *dt = desc.dt;
    }
    if (df) {
        *df = desc.df;
    }
    if (time && ndims < 5) {
        *time = 1;
    }
    if (desc.df == DF_MKT) {
        if (num) {
            *num = desc.dims[2];
        }
        if (numChannels) {
            *numChannels = desc.dims[1];
        }
        if (height) {
            *height = desc.dims[0];
        }
        if (width) {
            *width = 1;
        }
    } else if (desc.df == DF_MTK) {
        if (num) {
            *num = desc.dims[2];
        }
        if (numChannels) {
            *numChannels = desc.dims[0];
        }
        if (height) {
            *height = desc.dims[1];
        }
        if (width) {
            *width = 1;
        }
    } else if (desc.df == DF_NCTHW) {
        if (width) {
            *width = desc.dims[0];
        }
        if (height) {
            *height = desc.dims[1];
        }
        if (time) {
            *time = desc.dims[2];
        }
        if (numChannels) {
            *numChannels = desc.dims[3];
        }
        if (num) {
            *num = desc.dims[4];
        }
    } else {
        if (width) {
            *width = desc.dims[0];
        }
        if (height) {
            *height = (ndims > 1) ? desc.dims[1] : 1;
        }
        if (numChannels) {
            *numChannels = (ndims > 2) ? desc.dims[2] : 1;
        }
        if (num) {
            *num = (ndims > 3) ? desc.dims[3] : 1;
        }
    }
    return SUCCESS;
}

inline U32 tensorNumElements(TensorDesc desc)
{
    if (desc.nDims == 0) {
        return 0;
    }
    U32 ret = 1;
    for (U32 i = 0; i < desc.nDims; i++) {
        ret *= desc.dims[i];
    }
    return ret;
}

inline U32 tensorNumBytes(TensorDesc desc)
{
    if (desc.dt == DT_BIN01 || desc.dt == DT_BIN11) {
        return tensorNumElements(desc) / 8;
    } else {
        return tensorNumElements(desc) * bytesOf(desc.dt);
    }
}

inline U8 tensorIs1d(TensorDesc desc)
{
    return 1 == desc.nDims;
}

inline U8 tensorIs2d(TensorDesc desc)
{
    return 2 == desc.nDims;
}

inline U8 tensorIs3d(TensorDesc desc)
{
    return 3 == desc.nDims;
}

inline U8 tensorIs4d(TensorDesc desc)
{
    return 4 == desc.nDims;
}

inline U8 tensorIs5d(TensorDesc desc)
{
    return 5 == desc.nDims;
}

inline std::string tensorDesc2Str(TensorDesc desc)
{
    std::string descStr = "dt:" + std::to_string(desc.dt) + " df:" + std::to_string(desc.df) +
        " dims:" + std::to_string(desc.nDims);

    if (desc.nDims > 0) {
        descStr += "(";
    }
    for (I32 i = int(desc.nDims) - 1; i >= 0; i--) {
        descStr += std::to_string(desc.dims[i]);
        if (i > 0) {
            descStr += ",";
        } else {
            descStr += ")";
        }
    }

    return descStr;
}

inline int tensorDescIsValid(TensorDesc desc)
{
    if (desc.dt < 0 || desc.dt >= 10) {
        return 0;
    }

    if (desc.df < 0 || desc.df >= 30) {
        return 0;
    }

    if (desc.nDims > 6) {
        return 0;
    }

    for (U32 i = 0; i < desc.nDims; i++) {
        if (desc.dims[i] > INT_MAX) {
            return 0;
        }
    }

    return 1;
}

inline DataFormat getTensorDefaultDataFormat(int nDims)
{
    DataFormat df = DF_NORMAL;
    switch (nDims) {
        case 2:
            df = DF_NORMAL;
            break;
        case 3:
            df = DF_MTK;
            break;
        case 4:
            df = DF_NCHW;
            break;
        default:
            break;
    }
    return df;
}

inline std::vector<U32> calculateLocalIndex(U32 index, U32 *dims, U32 nDims)
{
    std::vector<U32> indexes(nDims);
    for (U32 i = 0; i < nDims; i++) {
        indexes[i] = index % dims[i];
        index /= dims[i];
    }
    return indexes;
}

inline U32 calculateGlobalIndex(U32 *indexes, U32 *dims, U32 nDims)
{
    U32 index = 0;
    for (int i = ((int)nDims) - 1; i >= 0; i--) {
        index = index * dims[i] + indexes[i];
    }
    return index;
}

void UNI_memcpy(void *dst, const void *src, int size);

void UNI_init(U32 num, DataType dt, F32 val, void *dst);

EE array_transpose(DataType dt,
    U32 *inputDims,
    const void *input,
    U32 *outputDims,
    void *output,
    U32 *transposeDims,
    int dimsNum);

void transformFromFloat(DataType dataType, float *src, void *dst, int num, float scale = 1.0);

void transformToFloat(DataType dataType, void *src, float *dst, int num, float scale = 1.0);

EE transformToNCHW(TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output);

EE transformToNHWC(TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output);

EE transformNCHWToNCHWC8(
    TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output);

EE transformNCHWC8ToNCHWC8ByGroup(
    TensorDesc inputDesc, const void *input, int group, TensorDesc outputDesc, void *output);

EE transposeFilter(TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output);
#endif
