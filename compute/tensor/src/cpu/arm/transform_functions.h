// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "types.h"
#ifdef _USE_FP32
#include "cpu/arm/fp32/convolution_winograd_transform.h"
#endif
#ifdef _USE_FP16
#include "cpu/arm/fp16/convolution_winograd_transform.h"
#endif

template <typename T, U32 N>
inline EE transformCNHWToNHWCNx(
    TensorDesc inputDesc, const T *input, TensorDesc outputDesc, T *output)
{
    if (input == NULL || output == NULL) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(inputDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    U32 oc = fc / N;
    U32 hwMax = fh * fw - 1;

    for (U32 o = 0; o < oc; o++) {
        for (U32 hw = 0; hw < fh * fw; hw++) {
            for (U32 c = 0; c < fn; c++) {
                for (U32 ox = 0; ox < N; ox++) {
                    output[o * fh * fw * fn * N + hw * fn * N + c * N + ox] =
                        input[c * fc * fh * fw + (o * N + ox) * fh * fw + hwMax - hw];
                }
            }
        }
    }
    if ((fc != oc * N) && (N == 16)) {
        for (U32 hw = 0; hw < fh * fw; hw++) {
            for (U32 c = 0; c < fn; c++) {
                for (U32 o8 = 0; o8 < 8; o8++) {
                    output[(oc * 16) * fh * fw * fn + hw * fn * 8 + c * 8 + o8] =
                        input[c * fc * fh * fw + (oc * 16 + o8) * fh * fw + hwMax - hw];
                }
            }
        }
    }
    return SUCCESS;
}

template <typename T>
inline EE transformCNHWToNCHWC8(
    TensorDesc inputDesc, const T *input, TensorDesc outputDesc, T *output)
{
    if (input == NULL || output == NULL) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(inputDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_REQUIREMENT(1 == fn);
    U32 ic = fc / 8;
    U32 hwMax = fh * fw - 1;
    for (U32 c = 0; c < ic; c++) {
        for (U32 hw = 0; hw < fh * fw; hw++) {
            for (U32 c8 = 0; c8 < 8; c8++) {
                output[c * fh * fw * 8 + hw * 8 + c8] = input[(c * 8 + c8) * fh * fw + hwMax - hw];
            }
        }
    }
    return SUCCESS;
}

template <typename T, U32 N>
inline EE transformCNHWToHWNCNx(
    TensorDesc inputDesc, const T *input, TensorDesc outputDesc, T *output)
{
    if (input == NULL || output == NULL) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(inputDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    const U32 hwMax = 8;
    for (U32 o = 0; o < fc / N; o++) {
        for (U32 c = 0; c < fn; c++) {
            U32 f_off_0 = c * fc * fh * fw + (o * N) * fh * fw;
            U32 f_off_1 = c * fc * fh * fw + (o * N + N / 2) * fh * fw;
            U32 ftm_off_0 = o * 36 * fn * N + c * N;
            U32 ftm_off_1 = o * 36 * fn * N + c * N + N / 2;
            T F[9][N / 2];
            T *F_ptr[9];
            T *Fw[36];

            for (U32 hw = 0; hw < 9; hw++) {
                for (U32 oo = 0; oo < N / 2; oo++) {
                    F[hw][oo] = input[f_off_0 + hwMax - hw + oo * fh * fw];
                }
                F_ptr[hw] = F[hw];
            }
            for (U32 hw = 0; hw < 36; hw++) {
                Fw[hw] = output + ftm_off_0 + hw * fn * N;
            }
            trans_W_4x4_3x3(Fw, F_ptr);
            for (U32 hw = 0; hw < 9; hw++) {
                for (U32 oo = 0; oo < N / 2; oo++) {
                    F[hw][oo] = input[f_off_1 + hwMax - hw + oo * fh * fw];
                }
                F_ptr[hw] = F[hw];
            }
            for (U32 hw = 0; hw < 36; hw++) {
                Fw[hw] = output + ftm_off_1 + hw * fn * N;
            }
            trans_W_4x4_3x3(Fw, F_ptr);
        }
    }
    U32 oc = (fc / 16) * 16;
    if ((oc != fc) && (N == 16)) {
        for (U32 c = 0; c < fn; c++) {
            U32 f_off_0 = c * fc * fh * fw + oc * fh * fw;
            U32 ftm_off_0 = oc * 36 * fn + c * 8;
            T F[9][8];
            T *F_ptr[9];
            T *Fw[36];
            for (U32 hw = 0; hw < 9; hw++) {
                for (U32 oo = 0; oo < 8; oo++) {
                    F[hw][oo] = input[f_off_0 + hwMax - hw + oo * fh * fw];
                }
                F_ptr[hw] = F[hw];
            }
            for (U32 hw = 0; hw < 36; hw++) {
                Fw[hw] = output + ftm_off_0 + hw * fn * 8;
            }
            trans_W_4x4_3x3(Fw, F_ptr);
        }
    }
    return SUCCESS;
}
