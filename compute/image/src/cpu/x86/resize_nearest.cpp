// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/image_x86.h"
#include "cpu/cpu_functions.h"

template <CoordinateTransMode coordinate_transformation_mode, RoundMode round_mode>
inline static EE resize_nearest_kernel_nchwc16(
    const TensorDesc &inputDesc, F32 *inArray, const TensorDesc &outputDesc, F32 *outArray)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    CHECK_REQUIREMENT(idf == DF_NCHWC16 || odf == DF_NCHW);
    CHECK_REQUIREMENT(odf == DF_NCHWC16 || odf == DF_NCHW);
    float r0_w = iw / (float)ow;
    float r0_h = ih / (float)oh;
    float r1_w = (iw - 1.0f) / (ow - 1.0f);
    float r1_h = (ih - 1.0f) / (oh - 1.0f);

    U32 ohow = oh * ow;
    U32 ihiw = ih * iw;
    U32 loop = on * ohow;

#ifdef _USE_AVX512_VNNI
    __m512i vindex = _mm512_set_epi32(ihiw * 15, ihiw * 14, ihiw * 13, ihiw * 12, ihiw * 11,
        ihiw * 10, ihiw * 9, ihiw * 8, ihiw * 7, ihiw * 6, ihiw * 5, ihiw * 4, ihiw * 3, ihiw * 2,
        ihiw, 0);

#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
    for (U32 l = 0; l < loop; ++l) {
        U32 n = l / ohow;
        U32 h = l % ohow / ow;
        U32 w = l % ohow % ow;

        I32 hC = round_d<round_mode>(coordinate_trans<coordinate_transformation_mode>(h, ih, oh, r0_h, r1_h));
        I32 wC = round_d<round_mode>(coordinate_trans<coordinate_transformation_mode>(w, iw, ow, r0_w, r1_w));

        // process edge pixel, linear
        hC = UNI_MIN((I32)ih - 1, UNI_MAX(0, hC));
        wC = UNI_MIN((I32)iw - 1, UNI_MAX(0, wC));

        if (idf == DF_NCHWC16) {
            for (U32 c = 0; c < oc; c += 16) {
                U32 output_idx = n * oc * ohow + c * ohow + h * ow * 16 + w * 16;
                U32 input_idx = n * ic * ihiw + c * ihiw + hC * iw * 16 + wC * 16;
                _mm512_storeu_ps(outArray + output_idx, _mm512_loadu_ps(inArray + input_idx));
            }
        } else if (odf == DF_NCHW) {
            for (U32 c = 0; c < oc; ++c) {
                U32 output_idx = n * oc * ohow + c * ohow + h * ow + w;
                U32 input_idx = n * ic * ihiw + c * ihiw + hC * iw + wC;
                outArray[output_idx] = inArray[input_idx];
            }
        } else if (odf == DF_NCHWC16) {
            for (U32 c = 0; c < oc; c += 16) {
                U32 output_idx = n * oc * ohow + c * ohow + h * ow * 16 + w * 16;
                U32 input_idx = n * ic * ihiw + c * ihiw + hC * iw + wC;
                outArray[output_idx] = inArray[input_idx];
                _mm512_storeu_ps(
                    outArray + output_idx, _mm512_i32gather_ps(vindex, inArray + input_idx, 4));
            }
        }
    }
#endif
    return SUCCESS;
}

template <CoordinateTransMode coordinate_transformation_mode, RoundMode round_mode>
inline static EE resize_nearest_kernel_nchwc8(
    const TensorDesc &inputDesc, F32 *inArray, const TensorDesc &outputDesc, F32 *outArray)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    CHECK_REQUIREMENT(idf == DF_NCHWC8 || idf == DF_NCHW);
    CHECK_REQUIREMENT(odf == DF_NCHWC8 || odf == DF_NCHW);
    float r0_w = iw / (float)ow;
    float r0_h = ih / (float)oh;
    float r1_w = (iw - 1.0f) / (ow - 1.0f);
    float r1_h = (ih - 1.0f) / (oh - 1.0f);

    U32 ohow = oh * ow;
    U32 ihiw = ih * iw;
    U32 loop = on * ohow;

    __m256i vindex =
        _mm256_set_epi32(ihiw * 7, ihiw * 6, ihiw * 5, ihiw * 4, ihiw * 3, ihiw * 2, ihiw, 0);

#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
    for (U32 l = 0; l < loop; ++l) {
        U32 n = l / ohow;
        U32 h = l % ohow / ow;
        U32 w = l % ohow % ow;

        I32 hC = round_d<round_mode>(coordinate_trans<coordinate_transformation_mode>(h, ih, oh, r0_h, r1_h));
        I32 wC = round_d<round_mode>(coordinate_trans<coordinate_transformation_mode>(w, iw, ow, r0_w, r1_w));

        // process edge pixel, linear
        hC = UNI_MIN((I32)ih - 1, UNI_MAX(0, hC));
        wC = UNI_MIN((I32)iw - 1, UNI_MAX(0, wC));

        if (idf == DF_NCHWC8) {
            for (U32 c = 0; c < oc; c += 8) {
                U32 output_idx = n * oc * ohow + c * ohow + h * ow * 8 + w * 8;
                U32 input_idx = n * ic * ihiw + c * ihiw + hC * iw * 8 + wC * 8;
                _mm256_storeu_ps(outArray + output_idx, _mm256_loadu_ps(inArray + input_idx));
            }
        } else if (odf == DF_NCHW) {
            for (U32 c = 0; c < oc; ++c) {
                U32 output_idx = n * oc * ohow + c * ohow + h * ow + w;
                U32 input_idx = n * ic * ihiw + c * ihiw + hC * iw + wC;
                outArray[output_idx] = inArray[input_idx];
            }
        } else if (odf == DF_NCHWC8) {
            for (U32 c = 0; c < oc; c += 8) {
                U32 output_idx = n * oc * ohow + c * ohow + h * ow * 8 + w * 8;
                U32 input_idx = n * ic * ihiw + c * ihiw + hC * iw + wC;
                outArray[output_idx] = inArray[input_idx];
                _mm256_storeu_ps(
                    outArray + output_idx, _mm256_i32gather_ps(inArray + input_idx, vindex, 4));
            }
        }
    }
    return SUCCESS;
}

template <CoordinateTransMode coordinate_transformation_mode, RoundMode round_mode>
inline static EE resize_nearest_kernel_nchwcx(
    const TensorDesc &inputDesc, F32 *inArray, const TensorDesc &outputDesc, F32 *outArray)
{
    if ((inputDesc.df == DF_NCHWC16) || (outputDesc.df == DF_NCHWC16)) {
        return resize_nearest_kernel_nchwc16<coordinate_transformation_mode, round_mode>(
            inputDesc, inArray, outputDesc, outArray);
    } else {
        return resize_nearest_kernel_nchwc8<coordinate_transformation_mode, round_mode>(
            inputDesc, inArray, outputDesc, outArray);
    }
}

template <CoordinateTransMode coordinate_transformation_mode>
inline static EE resize_nearest_kernel(const TensorDesc &inputDesc,
    F32 *inArray,
    ResizeParamSpec p,
    const TensorDesc &outputDesc,
    F32 *outArray)
{
    EE ret = SUCCESS;
    switch (p.round_mode) {
        case ROUND_CEIL: {
            resize_nearest_kernel_nchwcx<coordinate_transformation_mode, ROUND_CEIL>(
                inputDesc, inArray, outputDesc, outArray);
            break;
        }
        case ROUND_FLOOR: {
            resize_nearest_kernel_nchwcx<coordinate_transformation_mode, ROUND_FLOOR>(
                inputDesc, inArray, outputDesc, outArray);
            break;
        }
        case ROUND_PREFER_CEIL: {
            resize_nearest_kernel_nchwcx<coordinate_transformation_mode, ROUND_PREFER_CEIL>(
                inputDesc, inArray, outputDesc, outArray);
            break;
        }
        case ROUND_PREFER_FLOOR: {
            resize_nearest_kernel_nchwcx<coordinate_transformation_mode, ROUND_PREFER_FLOOR>(
                inputDesc, inArray, outputDesc, outArray);
            break;
        }
        default:
            UNI_ERROR_LOG("Resize currently not support this round mode.\n");
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

inline static EE resize_nearest_wrapper(const TensorDesc &inputDesc,
    F32 *inArray,
    const ResizeParamSpec &p,
    const TensorDesc &outputDesc,
    F32 *outArray)
{
    EE ret = SUCCESS;
    switch (p.trans_mode) {
        case COORDINATE_TRANS_HALF_PIXEL: {
            resize_nearest_kernel<COORDINATE_TRANS_HALF_PIXEL>(
                inputDesc, inArray, p, outputDesc, outArray);
            break;
        }
        case COORDINATE_TRANS_PYTORCH_HALF_PIXEL: {
            resize_nearest_kernel<COORDINATE_TRANS_PYTORCH_HALF_PIXEL>(
                inputDesc, inArray, p, outputDesc, outArray);
            break;
        }
        case COORDINATE_TRANS_ALIGN_CORNERS: {
            resize_nearest_kernel<COORDINATE_TRANS_ALIGN_CORNERS>(
                inputDesc, inArray, p, outputDesc, outArray);
            break;
        }
        case COORDINATE_TRANS_ASYMMETRIC: {
            resize_nearest_kernel<COORDINATE_TRANS_ASYMMETRIC>(
                inputDesc, inArray, p, outputDesc, outArray);
            break;
        }
        default:
            UNI_ERROR_LOG("Resize currently not support this coordinate transformation mode.\n");
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE resize_nearest_x86(
    TensorDesc inputDesc, void *input, ResizeParamSpec p, TensorDesc outputDesc, void *output)
{
    EE ret = NOT_SUPPORTED;
    switch (inputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = resize_nearest_wrapper(inputDesc, (F32 *)input, p, outputDesc, (F32 *)output);
            break;
        }
#endif
        default:
            break;
    }
    return ret;
}
