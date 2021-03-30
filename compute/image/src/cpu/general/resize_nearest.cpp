// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/general/image_general.h"

template <typename IT, typename OT>
EE resize_nearest(TensorDesc inputDesc, IT *inArray, TensorDesc outputDesc, OT *outArray)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if (idf == DF_NCHWC8) {
        CHECK_STATUS(from_nchwc8_to_nchw<IT>(&inputDesc, inArray));
    }
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    if (idf != DF_NCHW && idf != DF_RGB) {
        CHECK_STATUS(NOT_MATCH);
    }

    // naive implement
    float widthRate = iw * 1.0 / ow;
    float heightRate = ih * 1.0 / oh;

    for (U32 n = 0; n < on; n++) {
        for (U32 c = 0; c < oc; c++) {
            U32 offSet_in = n * (ic * ih * iw) + c * (ih * iw);
            U32 offSet_out = n * (oc * oh * ow) + c * (oh * ow);  // offSet of output
            for (U32 h = 0; h < oh; h++) {
                for (U32 w = 0; w < ow; w++) {
                    int srcX = (h * widthRate - (int)(h * widthRate) > 0.5)
                        ? ((int)(h * widthRate) + 1)
                        : (int)(h * widthRate);
                    int srcY = (w * heightRate - (int)(w * heightRate) > 0.5)
                        ? ((int)(w * heightRate) + 1)
                        : (int)(w * heightRate);
                    outArray[offSet_out + h * ow + w] = (OT)(inArray[offSet_in + srcX * iw + srcY]);
                }
            }
        }
    }

    if (odf == DF_NCHWC8) {
        outputDesc.df = DF_NCHW;
        CHECK_STATUS(from_nchw_to_nchwc8<OT>(&outputDesc, outArray));
    }
    return SUCCESS;
}

EE resize_nearest_general(TensorDesc inputDesc, void *input, TensorDesc outputDesc, void *output)
{
    EE ret = NOT_SUPPORTED;
    switch (inputDesc.dt) {
#ifdef __aarch64__
        case DT_F16: {
            ret = resize_nearest<F16, F16>(inputDesc, (F16 *)input, outputDesc, (F16 *)output);
            break;
        }
#endif
#ifdef _USE_FP32
        case DT_F32: {
            ret = resize_nearest<F32, F32>(inputDesc, (F32 *)input, outputDesc, (F32 *)output);
            break;
        }
#endif
        case DT_U8: {
#ifdef __aarch64__
            if (DT_F16 == outputDesc.dt) {
                ret = resize_nearest<U8, F16>(inputDesc, (U8 *)input, outputDesc, (F16 *)output);
            }
#endif
#ifdef _USE_FP32
            if (DT_F32 == outputDesc.dt) {
                ret = resize_nearest<U8, F32>(inputDesc, (U8 *)input, outputDesc, (F32 *)output);
            }
#endif
            break;
        }
        default:
            break;
    }
    return ret;
}
