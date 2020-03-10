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
#include <stdio.h>
#include <limits.h>
#include <string>
#include "type.h"
#include "error.h"

    typedef enum {
        DF_NCHW,
        DF_NCHWN16, //vectorize for N=16, for filter
        DF_NCHWC8, //vectorize for C=8, for input and output
        DF_HWNCN16, //vectorize for N=16, for filter in winograd
        DF_NHWCN16, // im2col + GEMM, for filter
        DF_NHWCN8, //vectorize for N=8, not used
        DF_HWNCN8C4, //int8 filter for winograd
        DF_NCHWN8C4, //int8 im2col + GEMM, for filter
        DF_NCHWN8HW4, //int8 im2col + GEMM in the first layer, for filter
        DF_NCHWN16C8, //bnn im2col + GEMM, for filter
        DF_TRANSPOSE, //vectorize for COL_MAJOR
        DF_NORMAL,  //vectorize for ROW_MAJOR
        DF_MTK, // LSTM input, T: step, M: batch, K: x_dim
        DF_NK, // MMM/MVM filter, N: col_num, K: row_num
        DF_NKN32, // MMM/MVM filter, vectorized for N=32
        DF_8NK, // LSTM MMM/MVM filter, with 8 matrix
        DF_CHW_NC, // dw_conv, CHW means dw part, NC means pw part
        DF_CHWC8_NCN16, // dw_conv, vectorized for C8 and N16
        DF_CHWC8_NCN8C4, // int8 dw_conv, vectorized for C4 and N8
        DF_NCWHC4, //ocl mali input and output
        DF_NCHWC3, //ocl mali support input rgb
        DF_NHWC, //ocl mali support input/output
        DF_NCHWN4C4, //ocl mali conv filter
        DF_NCWHN4C4, //ocl mali fc   filter
        DF_NHWCN4,  //ocl mali filter
        DF_CHWC8_NCN8, // fp32 dw_conv, vectorized for C8 and N8
        DF_RGB,
        DF_HWNCN8  // fp32 filter for winograd
    } DataFormat;

    typedef struct {
        DataType dt;
        DataFormat df;
        U32 nDims = 0;
        U32 dims[6] = {0};
    } TensorDesc;

    /**
     * @param num the number of filter or image, not count for the last dim for vectorize
     *
     **/
    inline TensorDesc tensor4df(DataType dt, DataFormat df, U32 num, U32 numChannels, U32 height, U32 width) {
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

    inline EE tensor1dGet(TensorDesc desc, DataType* dt, U32* len)
    {
        if (nullptr == len || nullptr == dt) {
            return NULL_POINTER;
        }
        if (1 != desc.nDims) {
            return NOT_MATCH;
        }

        *dt = desc.dt;
        *len = desc.dims[0];
        return SUCCESS;
    }

    inline EE tensor2dfGet(TensorDesc desc, DataType* dt, DataFormat *df, U32* numRows, U32* numColumns)
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

    inline EE tensor2dGet(TensorDesc desc, DataType* dt, U32* numRows, U32* numColumns)
    {
        if (nullptr == numColumns || nullptr == numRows || nullptr == dt) {
            return NULL_POINTER;
        }
        if (2 != desc.nDims) {
            return NOT_MATCH;
        }

        *dt = desc.dt;
        *numColumns = desc.dims[0];
        *numRows = desc.dims[1];
        return SUCCESS;
    }

    inline EE tensor3dGet(TensorDesc desc, DataType* dt, DataFormat *df, U32* numChannels, U32* height, U32* width) 
    {
        if (nullptr == numChannels || nullptr == height || nullptr == width || nullptr == dt || nullptr == df) {
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

    inline EE tensor4dGet(TensorDesc desc, DataType* dt, DataFormat *df, U32* num, U32* numChannels, U32* height, U32* width)
    {
        if (nullptr == num || nullptr == numChannels || nullptr == height || nullptr == width || nullptr == dt || nullptr == df) {
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
 
    inline EE tensorSelectGet(TensorDesc desc, DataType* dt, DataFormat *df, U32* num, U32* numChannels, U32* height, U32* width)
    {
        if (dt)          *dt = desc.dt;
        if (df)          *df = desc.df;
        if (width)       *width = desc.dims[0];
        if (height)      *height = desc.dims[1];
        if (numChannels) *numChannels = desc.dims[2];
        if (num)         *num = desc.dims[3];
        return SUCCESS;
    }

    inline U32 tensorNumElements(TensorDesc desc)
    {
        if (desc.nDims == 0) return 0;
        U32 ret = 1;
        if (1 <= desc.nDims) ret *= desc.dims[0];
        if (2 <= desc.nDims) ret *= desc.dims[1];
        if (3 <= desc.nDims) ret *= desc.dims[2];
        if (4 <= desc.nDims) ret *= desc.dims[3];
        if (5 <= desc.nDims) ret *= desc.dims[4];

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

    inline U8 tensorIs1d(TensorDesc desc) {
        return 1 == desc.nDims;
    }

    inline U8 tensorIs2d(TensorDesc desc) {
        return 2 == desc.nDims;
    }

    inline U8 tensorIs3d(TensorDesc desc) {
        return 3 == desc.nDims;
    }

    inline U8 tensorIs4d(TensorDesc desc) {
        return 4 == desc.nDims;
    }

    inline std::string tensorDesc2Str(TensorDesc desc)
    {
        char buff[128];
        snprintf(buff, sizeof(buff), "dt:%d df:%d dims:%d", desc.dt, desc.df, desc.nDims);
        std::string descStr = buff;

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
        if (desc.dt < 0 || desc.dt >= 10)
            return 0;

        if (desc.df < 0 || desc.df >= 30)
            return 0;

        if (desc.nDims > 6)
            return 0;

        for (U32 i = 0; i < desc.nDims; i++) {
            if (desc.dims[i] > INT_MAX)
               return 0;
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
#endif
