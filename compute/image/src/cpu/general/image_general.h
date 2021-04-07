// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_IMAGE_GENERAL
#define _H_IMAGE_GENERAL

#include "error.h"
#include "tensor_desc.h"

EE resize_bilinear_general(TensorDesc inputDesc, void *input, TensorDesc outputDesc, void *output);

EE resize_nearest_general(TensorDesc inputDesc, void *input, TensorDesc outputDesc, void *output);

template <typename T>
inline EE from_nchwc8_to_nchw(TensorDesc *desc, T *data)
{
    if (desc == nullptr || data == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }

    DataType idt;
    DataFormat idf;
    U32 in, ic, ih, iw;
    CHECK_STATUS(tensor4dGet(*desc, &idt, &idf, &in, &ic, &ih, &iw));
    if (idf != DF_NCHWC8) {
        CHECK_STATUS(NOT_MATCH);
    }

    *desc = tensor4df(idt, DF_NCHW, in, ic, ih, iw);

    T *tmp = (T *)malloc(tensorNumBytes(*desc));
    ic /= 8;
    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c++) {
            for (U32 hw = 0; hw < ih * iw; hw++) {
                for (U32 c8 = 0; c8 < 8; c8++) {
                    tmp[n * ic * 8 * ih * iw + (c * 8 + c8) * ih * iw + hw] =
                        data[n * ic * ih * iw * 8 + c * ih * iw * 8 + hw * 8 + c8];
                }
            }
        }
    }
    memcpy(data, tmp, tensorNumBytes(*desc));
    free(tmp);
    return SUCCESS;
}

template <typename T>
inline EE from_nchw_to_nchwc8(TensorDesc *desc, T *data)
{
    if (desc == nullptr || data == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }

    DataType idt;
    DataFormat idf;
    U32 in, ic, ih, iw;
    CHECK_STATUS(tensor4dGet(*desc, &idt, &idf, &in, &ic, &ih, &iw));
    if (idf != DF_NCHW) {
        CHECK_STATUS(NOT_MATCH);
    }

    *desc = tensor4df(idt, DF_NCHWC8, in, ic, ih, iw);

    T *tmp = (T *)malloc(tensorNumBytes(*desc));
    ic /= 8;
    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c++) {
            for (U32 hw = 0; hw < ih * iw; hw++) {
                for (U32 c8 = 0; c8 < 8; c8++) {
                    tmp[n * ic * ih * iw * 8 + c * ih * iw * 8 + hw * 8 + c8] =
                        data[n * ic * 8 * ih * iw + (c * 8 + c8) * ih * iw + hw];
                }
            }
        }
    }
    memcpy(data, tmp, tensorNumBytes(*desc));
    free(tmp);
    return SUCCESS;
}
#endif
