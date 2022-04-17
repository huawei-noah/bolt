// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TRANSFORM_FUNCTIONS
#define _H_TRANSFORM_FUNCTIONS
#ifdef _USE_FP32
#include "cpu/arm/fp32/convolution_winograd_transform.h"
#endif
#ifdef _USE_FP16
#include "cpu/arm/fp16/convolution_winograd_transform.h"
#endif
#include "thread_affinity.h"

#define _USE_CACHE
#ifdef _USE_CACHE
#include <map>
#include <vector>
extern std::map<std::string, std::vector<U32>> arm_cache;
#endif

template <typename T, U32 N>
inline EE transformCNHWToNHWCNx(
    TensorDesc inputDesc, const T *input, DataFormat odf, TensorDesc *outputDesc, T *output)
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
    for (U32 o = 0, out_id = 0; o < oc; o++) {
        for (U32 hw = 0; hw < fh * fw; hw++) {
            for (U32 c = 0; c < fn; c++) {
                for (U32 ox = 0; ox < N; ox++, out_id++) {
                    U32 in_id = (c * fc + (o * N + ox)) * fh * fw + hwMax - hw;
                    output[out_id] = input[in_id];
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
    *outputDesc = tensor4df(fdt, odf, fc, fn, fh, fw);
    return SUCCESS;
}

template <typename T>
inline EE transformCNHWToNCHWC8(
    TensorDesc inputDesc, const T *input, DataFormat odf, TensorDesc *outputDesc, T *output)
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
    for (U32 c = 0, out_id = 0; c < ic; c++) {
        for (U32 hw = 0; hw < fh * fw; hw++) {
            for (U32 c8 = 0; c8 < 8; c8++, out_id++) {
                U32 in_id = (c * 8 + c8) * fh * fw + hwMax - hw;
                output[out_id] = input[in_id];
            }
        }
    }
    *outputDesc = tensor4df(fdt, odf, fn, fc, fh, fw);
    return SUCCESS;
}

template <typename T, U32 N>
inline EE transformCNHWToHWNCNx(
    TensorDesc inputDesc, const T *input, DataFormat odf, TensorDesc *outputDesc, T *output)
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
    *outputDesc = tensor4df(fdt, odf, fc, fn, 6, 6);
    return SUCCESS;
}

template <typename T, U32 CAlignSize>
inline T *convolution_input_padding_per_channel(
    U32 n, U32 ic, U32 it, U32 ih, U32 iw, const ConvolutionParamSpec &p, T *src, T *dst)
{
    U32 it_pad = it + p.pad_before + p.pad_after;
    U32 ih_pad = ih + p.pad_top + p.pad_bottom;
    U32 iw_pad = iw + p.pad_left + p.pad_right;
    T *inArray_pad;
    if (p.pad_before == 0 && p.pad_after == 0 && p.pad_top == 0 && p.pad_bottom == 0 &&
        p.pad_left == 0 && p.pad_right == 0) {
        T *inArray_mov = src + n * ic * it * ih * iw * CAlignSize;
        inArray_pad = inArray_mov;
    } else {
        inArray_pad = dst;
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
        for (U32 c = 0; c < ic; c++) {
            T *inArray_mov = src + (n * ic + c) * it * ih * iw * CAlignSize;
            T *inArray_pad_mov = inArray_pad + c * it_pad * ih_pad * iw_pad * CAlignSize;
            UNI_MEMSET(inArray_pad_mov, 0, p.pad_before * ih_pad * iw_pad * CAlignSize * sizeof(T));
            inArray_pad_mov += p.pad_before * ih_pad * iw_pad * CAlignSize;
            for (U32 t = p.pad_before; t < it_pad - p.pad_after; t++) {
                UNI_MEMSET(inArray_pad_mov, 0, p.pad_top * iw_pad * CAlignSize * sizeof(T));
                inArray_pad_mov += p.pad_top * iw_pad * CAlignSize;
                for (U32 h = p.pad_top; h < ih_pad - p.pad_bottom; h++) {
                    UNI_MEMSET(inArray_pad_mov, 0, p.pad_left * CAlignSize * sizeof(T));
                    inArray_pad_mov += p.pad_left * CAlignSize;
                    UNI_MEMCPY(inArray_pad_mov, inArray_mov, iw * CAlignSize * sizeof(T));
                    inArray_pad_mov += iw * CAlignSize;
                    inArray_mov += iw * CAlignSize;
                    UNI_MEMSET(inArray_pad_mov, 0, p.pad_right * CAlignSize * sizeof(T));
                    inArray_pad_mov += p.pad_right * CAlignSize;
                }
                UNI_MEMSET(inArray_pad_mov, 0, p.pad_bottom * iw_pad * CAlignSize * sizeof(T));
                inArray_pad_mov += p.pad_bottom * iw_pad * CAlignSize;
            }
            UNI_MEMSET(inArray_pad_mov, 0, p.pad_after * ih_pad * iw_pad * CAlignSize * sizeof(T));
            inArray_pad_mov += p.pad_after * ih_pad * iw_pad * CAlignSize;
        }
    }
    return inArray_pad;
}

template <U32 TileSize, U32 CAlignSize>
inline void convolution_padding_input_offset(const U32 &ih_pad,
    const U32 &iw_pad,
    const ConvolutionParamSpec &p,
    const U32 &oh,
    const U32 &ow,
    const U32 &output_hw_start,
    U32 *input_offset)
{
    U32 ohow = oh * ow;
    for (U32 id = 0; id < TileSize; id++) {
        U32 thw_id = output_hw_start + id;
        U32 hw_id = thw_id % ohow;
        U32 iw_id = hw_id % ow * p.stride_w;
        U32 ih_id = hw_id / ow * p.stride_h;
        U32 it_id = thw_id / ohow * p.stride_t;
        input_offset[id] = ((it_id * ih_pad + ih_id) * iw_pad + iw_id) * CAlignSize;
    }
}

template <typename T, U32 TileSize, U32 CAlignSize>
inline void convolution_input_pack(const U32 &ic,
    const U32 &it_pad,
    const U32 &ih_pad,
    const U32 &iw_pad,
    const ConvolutionParamSpec &p,
    const U32 &ft,
    const U32 &fh,
    const U32 &fw,
    T *src,
    U32 *padding_input_offset,
    T *dst)
{
    for (U32 c = 0; c < ic; c++) {
        for (U32 ft_idx = 0; ft_idx < ft; ft_idx++) {
            for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                    T *in_hw = src +
                        (((c * it_pad + ft_idx * p.dilatedRate_t) * ih_pad +
                             fh_idx * p.dilatedRate_h) *
                                iw_pad +
                            p.dilatedRate_w * fw_idx) *
                            CAlignSize;
                    T *in_pack_hw = dst +
                        (((ft_idx * fh + fh_idx) * fw + fw_idx) * ic + c) * CAlignSize * TileSize;
                    for (U32 i = 0, id = 0; i < CAlignSize; i++) {
                        for (U32 j = 0; j < TileSize; j++, id++) {
                            in_pack_hw[id] = in_hw[padding_input_offset[j] + i];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
inline void convolution_nchwc8_input_pack_tile1(const U32 &ic,
    const U32 &it_pad,
    const U32 &ih_pad,
    const U32 &iw_pad,
    const ConvolutionParamSpec &p,
    const U32 &ft,
    const U32 &fh,
    const U32 &fw,
    const U32 &ot,
    const U32 &oh,
    const U32 &ow,
    T *src,
    U32 hw,
    T *dst)
{
    const int TileSize = 1;
    U32 padding_input_offset[TileSize];
    convolution_padding_input_offset<1, 8>(ih_pad, iw_pad, p, oh, ow, hw, padding_input_offset);
    convolution_input_pack<T, 1, 8>(
        ic, it_pad, ih_pad, iw_pad, p, ft, fh, fw, src, padding_input_offset, dst);
}

template <typename T, U32 TileSize>
inline void convolution_nchw_input_pack(const U32 &ic,
    const U32 &it_pad,
    const U32 &ih_pad,
    const U32 &iw_pad,
    const ConvolutionParamSpec &p,
    const U32 &ft,
    const U32 &fh,
    const U32 &fw,
    const U32 &ot,
    const U32 &oh,
    const U32 &ow,
    T *src,
    const U32 &hw,
    T *dst)
{
    U32 padding_input_offset[TileSize];
    convolution_padding_input_offset<TileSize, 1>(
        ih_pad, iw_pad, p, oh, ow, hw, padding_input_offset);
    convolution_input_pack<T, TileSize, 1>(
        ic, it_pad, ih_pad, iw_pad, p, ft, fh, fw, src, padding_input_offset, dst);
}

#ifdef _USE_CACHE
template <I32 TileSize, I32 CAlignSize>
inline U32 *get_convolution_padding_input_offset(const std::string &key,
    const U32 &ih_pad,
    const U32 &iw_pad,
    const ConvolutionParamSpec &p,
    const U32 &oh,
    const U32 &ow,
    const I32 &ohow)
{
    U32 *ret;
#ifdef _THREAD_SAFE
    pthread_mutex_lock(&uniThreadMutex);
#endif
    if (arm_cache.find(key) == arm_cache.end()) {
        std::vector<U32> offset(ohow);
        for (I32 hw = 0; hw < ohow - TileSize + 1; hw += TileSize) {
            convolution_padding_input_offset<TileSize, CAlignSize>(
                ih_pad, iw_pad, p, oh, ow, hw, offset.data() + hw);
        }
        arm_cache[key] = offset;
    }
    ret = arm_cache[key].data();
#ifdef _THREAD_SAFE
    pthread_mutex_unlock(&uniThreadMutex);
#endif
    return ret;
}
#endif

template <typename T, U32 N>
inline EE transformNCHWToNHWCNx(
    TensorDesc inputDesc, const T *input, DataFormat odf, TensorDesc *outputDesc, T *output)
{
    if (input == NULL || output == NULL) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, ft, fh, fw;
    if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
        ft = 1;
        *outputDesc = tensor4df(fdt, odf, fn, fc, fh, fw);
    } else if (tensorIs5d(inputDesc)) {
        CHECK_STATUS(tensor5dGet(inputDesc, &fdt, &fdf, &fn, &fc, &ft, &fh, &fw));
        *outputDesc = tensor5df(fdt, odf, fn, fc, ft, fh, fw);
    } else {
        return NOT_SUPPORTED;
    }
    U32 fthw = ft * fh * fw;
    U32 oc = fn / N;
    for (U32 o = 0, out_id = 0; o < oc; o++) {
        for (U32 hw = 0; hw < fthw; hw++) {
            for (U32 c = 0; c < fc; c++) {
                for (U32 o16 = 0; o16 < N; o16++, out_id++) {
                    U32 in_id = ((o * N + o16) * fc + c) * fthw + hw;
                    output[out_id] = input[in_id];
                }
            }
        }
    }
    if (fn != oc * N && N == 16) {
        for (U32 hw = 0; hw < fthw; hw++) {
            for (U32 c = 0; c < fc; c++) {
                for (U32 o8 = 0; o8 < 8; o8++) {
                    U32 in_id = ((oc * 16 + o8) * fc + c) * fthw + hw;
                    U32 out_id = (((oc * 2) * fthw + hw) * fc + c) * 8 + o8;
                    output[out_id] = input[in_id];
                }
            }
        }
    }
    return SUCCESS;
}

template <typename T, U32 N>
inline EE transformNCHWToNCHWNx(
    TensorDesc inputDesc, const T *input, DataFormat odf, TensorDesc *outputDesc, T *output)
{
    if (input == NULL || output == NULL) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, ft, fh, fw;
    if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
        ft = 1;
        *outputDesc = tensor4df(fdt, odf, fn, fc, fh, fw);
    } else if (tensorIs5d(inputDesc)) {
        CHECK_STATUS(tensor5dGet(inputDesc, &fdt, &fdf, &fn, &fc, &ft, &fh, &fw));
        *outputDesc = tensor5df(fdt, odf, fn, fc, ft, fh, fw);
    } else {
        return NOT_SUPPORTED;
    }
    U32 fcthw = fc * ft * fh * fw;
    U32 oc = fn / N;
    for (U32 o = 0, out_id = 0; o < oc; o++) {
        for (U32 chw = 0; chw < fcthw; chw++) {
            for (U32 o16 = 0; o16 < N; o16++, out_id++) {
                U32 in_id = (o * N + o16) * fcthw + chw;
                output[out_id] = input[in_id];
            }
        }
    }
    return SUCCESS;
}

template <typename T, U32 N>
inline EE transformNCHWToHWNCNx(
    TensorDesc inputDesc, const T *input, DataFormat odf, TensorDesc *outputDesc, T *output)
{
    if (input == NULL || output == NULL) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(inputDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    for (U32 o = 0; o < fn / N; o++) {
        for (U32 c = 0; c < fc; c++) {
            U32 f_off_0 = (o * N) * fc * fh * fw + c * fh * fw;
            U32 f_off_1 = (o * N + N / 2) * fc * fh * fw + c * fh * fw;
            U32 ftm_off_0 = o * 36 * fc * N + c * N;
            U32 ftm_off_1 = o * 36 * fc * N + c * N + N / 2;
            T F[9][N / 2];
            T *F_ptr[9];
            T *Fw[36];
            for (U32 hw = 0; hw < 9; hw++) {
                for (U32 oo = 0; oo < N / 2; oo++) {
                    F[hw][oo] = input[f_off_0 + hw + oo * fc * fh * fw];
                }
                F_ptr[hw] = F[hw];
            }
            for (U32 hw = 0; hw < 36; hw++) {
                Fw[hw] = output + ftm_off_0 + hw * fc * N;
            }
            trans_W_4x4_3x3(Fw, F_ptr);
            for (U32 hw = 0; hw < 9; hw++) {
                for (U32 oo = 0; oo < N / 2; oo++) {
                    F[hw][oo] = input[f_off_1 + hw + oo * fc * fh * fw];
                }
                F_ptr[hw] = F[hw];
            }
            for (U32 hw = 0; hw < 36; hw++) {
                Fw[hw] = output + ftm_off_1 + hw * fc * N;
            }
            trans_W_4x4_3x3(Fw, F_ptr);
        }
    }
    U32 oc = (fn / 16) * 16;
    if (oc != fn && N == 16) {
        for (U32 c = 0; c < fc; c++) {
            U32 f_off_0 = oc * fc * fh * fw + c * fh * fw;
            U32 ftm_off_0 = oc * 36 * fc + c * 8;
            T F[9][8];
            T *F_ptr[9];
            T *Fw[36];
            for (U32 hw = 0; hw < 9; hw++) {
                for (U32 oo = 0; oo < 8; oo++) {
                    F[hw][oo] = input[f_off_0 + hw + oo * fc * fh * fw];
                }
                F_ptr[hw] = F[hw];
            }
            for (U32 hw = 0; hw < 36; hw++) {
                Fw[hw] = output + ftm_off_0 + hw * fc * 8;
            }
            trans_W_4x4_3x3(Fw, F_ptr);
        }
    }
    *outputDesc = tensor4df(fdt, odf, fn, fc, 6, 6);
    return SUCCESS;
}
#endif
