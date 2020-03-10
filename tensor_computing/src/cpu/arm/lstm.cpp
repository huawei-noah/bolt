// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <string.h>

#include "cpu/arm/tensor_computing_arm.h"
#ifdef _USE_FP32
#include "cpu/arm/fp32/tensor_computing_fp32.h"
#endif
#ifdef _USE_FP16
#include "cpu/arm/fp16/tensor_computing_fp16.h"
#endif

template<typename T>
EE lstm_transform_filter(TensorDesc filterDesc, const T* filterArray, TensorDesc *ftmDesc, T* ftmArray, U32 x_dim, U32 h_dim)
{
    if (nullptr == filterArray || nullptr == ftmArray || nullptr == ftmArray)
        CHECK_STATUS(NULL_POINTER);
    DataType fdt;
    DataFormat fdf;
    U32 fn, fk, ftm_n, ftm_k;
    CHECK_STATUS(tensor2dfGet(filterDesc, &fdt, &fdf, &fn, &fk));
    if(fn != 4*h_dim || fk != (x_dim + h_dim))
        CHECK_STATUS(NOT_MATCH);
    
    switch(fdf) {
        case DF_NKN32: {
            // everything is ready
            ftm_n = fn;
            ftm_k = fk;
            break;
        }
        case DF_NK: {
            // NK => NKN32
            if (fn % 32 != 0) {
                return NOT_MATCH;
            }
            ftm_n = fn/32;
            ftm_k = fk;
            for (U32 n = 0; n < ftm_n; n++) {
                for (U32 k = 0; k < ftm_k; k++) {
                    for (U32 n32 = 0; n32 < 32; n32++) {
                        ftmArray[n*ftm_k*32 + k*32 + n32] = filterArray[(n*32+n32)*ftm_k + k];
                    }
                }
            }
            break;
        }
        case DF_8NK: {
            // combine 8 matrix into 1 NK => NKN32
            // assume the order of 8 matrix is: h_I, h_F, h_O, h_G, x_I, x_F, x_O, x_G
            if (h_dim % 8 != 0) {
                return NOT_MATCH;
            }
            ftm_n = 4*h_dim/32;
            ftm_k = h_dim + x_dim;
            for (U32 n = 0; n < ftm_n; n++) {
                for (U32 hk = 0; hk < h_dim; hk++) {
                    for (U32 n32 = 0; n32 < 32; n32++) {
                        ftmArray[n*ftm_k*32 + hk*32 + n32] = filterArray[(n*32+n32)*h_dim + hk];
                    }
                }
                for (U32 xk = 0; xk < x_dim; xk++) {
                    for (U32 n32 = 0; n32 < 32; n32++) {
                        ftmArray[n*ftm_k*32 + (h_dim+xk)*32 + n32] = filterArray[ftm_n*32*h_dim + (n*32+n32)*x_dim + xk];
                    }
                }
            }
            break;
        }
        default:
            return NOT_MATCH;
    }
    *ftmDesc = tensor2df(fdt, DF_NKN32, fn, fk);
    return SUCCESS;
}

EE lstm_transform_filter_arm(TensorDesc filterDesc, const void* filterArray, TensorDesc *ftmDesc, void* ftmArray, U32 x_dim, U32 h_dim)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = lstm_transform_filter<F32>(filterDesc, (const F32*)filterArray, ftmDesc, (F32*)ftmArray, x_dim, h_dim);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = lstm_transform_filter<F16>(filterDesc, (const F16*)filterArray, ftmDesc, (F16*)ftmArray, x_dim, h_dim);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE lstm_transform_filter_bytes_arm(TensorDesc filterDesc, U32* bytes)
{
    if (nullptr == bytes)
        CHECK_STATUS(NULL_POINTER);
    *bytes = tensorNumBytes(filterDesc);
    return SUCCESS;
}

EE lstmcell_infer_forward_tmp_bytes_arm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc, LSTMDesc lstmDesc, U32 *bytes)
{
    UNUSED(filterDesc);
    UNUSED(outputDesc);
    if (nullptr == bytes)
        CHECK_STATUS(NULL_POINTER);
    DataType idt;
    DataFormat idf;
    U32 batch, xDim;
    CHECK_STATUS(tensor2dfGet(inputDesc, &idt, &idf, &batch, &xDim));
    U32 hDim = lstmDesc.numOutput;
    *bytes = (hDim + xDim + hDim * 4) * bytesOf(idt);
    return SUCCESS;
}

EE lstm_infer_forward_tmp_bytes_arm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc, LSTMDesc lstmDesc, U32 *bytes)
{
    UNUSED(filterDesc);
    UNUSED(outputDesc);
    if (nullptr == bytes)
        CHECK_STATUS(NULL_POINTER);
    DataType idt;
    DataFormat idf;
    U32 batch, step, xDim;
    CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &batch, &step, &xDim));
    U32 hDim = lstmDesc.numOutput;
    TensorDesc xDesc = tensor2df(idt, DF_NORMAL, batch, xDim);
    CHECK_STATUS(lstmcell_infer_forward_tmp_bytes_arm(xDesc, filterDesc, outputDesc, lstmDesc, bytes));
    *bytes += batch * (hDim + hDim) * bytesOf(idt);
    return SUCCESS;
}

EE lstmcell_arm(TensorDesc xDesc, const void* currentX,
    TensorDesc filterDesc, const void* filter,
    TensorDesc biasDesc, const void* bias,
    void *state,
    U32 tmpBytes, void *tmp,
    LSTMDesc lstmDesc, U32 batchStrideX, U32 batchStrideH,
    TensorDesc hDesc, void* output)
{
    EE ret = SUCCESS;
    switch (xDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = lstmcell_fp32(xDesc, currentX, 
                                filterDesc, filter,
                                biasDesc, bias,
                                state,
                                tmpBytes, tmp,
                                lstmDesc, batchStrideX, batchStrideH,
                                hDesc, output);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = lstmcell_fp16(xDesc, currentX,
                                filterDesc, filter,
                                biasDesc, bias,
                                state,
                                tmpBytes, tmp,
                                lstmDesc, batchStrideX, batchStrideH,
                                hDesc, output);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE lstm_arm(TensorDesc inputDesc, const void* input,
    TensorDesc filterDesc, const void* filter,
    TensorDesc biasDesc, const void* bias,
    U32 tmpBytes, void* tmp,
    LSTMDesc lstmDesc,
    TensorDesc outputDesc, void* output)
{
    UNUSED(outputDesc);

    if (nullptr == input
        || nullptr == filter
        || nullptr == bias
        || nullptr == tmp
        || nullptr == output)
        CHECK_STATUS(NULL_POINTER);

    DataType idt;
    DataFormat idf;
    U32 batch, step, xDim;
    CHECK_STATUS(tensor3dGet(inputDesc,  &idt, &idf, &batch, &step, &xDim));
    U32 hDim = lstmDesc.numOutput;

    U8 *cellState = (U8*)tmp;
    U8 *tmpArray = cellState + batch * (hDim + hDim) * bytesOf(idt);
    memset(cellState, 0, batch * (hDim + hDim) * bytesOf(idt));
    U32 batchStrideX = step * xDim;
    U32 batchStrideH = step * hDim;
    TensorDesc xDesc = tensor2df(idt, DF_NORMAL, batch, xDim);
    TensorDesc hDesc = tensor2df(idt, DF_NORMAL, batch, hDim);

    for (U32 t = 0; t < step; t++) {
        const U8* currentX = (const U8*)input + t * xDim * bytesOf(idt);
        U8 *currentH = (U8*)output + t * hDim * bytesOf(idt);
        CHECK_STATUS(lstmcell_arm(xDesc, currentX,
              filterDesc, filter,
              biasDesc, bias,
              cellState,
              tmpBytes, tmpArray,
              lstmDesc, batchStrideX, batchStrideH,
              hDesc, currentH));
    }
    return SUCCESS;
}
