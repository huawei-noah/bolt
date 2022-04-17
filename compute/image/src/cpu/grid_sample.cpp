// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/image_cpu.h"
#include "uni.h"

EE grid_sample_infer_output_size_cpu(
    TensorDesc inputDesc, TensorDesc gridDesc, TensorDesc *outputDesc)
{
    *outputDesc = inputDesc;
    outputDesc->dims[0] = gridDesc.dims[1];
    outputDesc->dims[1] = gridDesc.dims[2];
    CHECK_REQUIREMENT(gridDesc.dims[0] == inputDesc.nDims - 2);
    return SUCCESS;
}

static inline float denormalize(float n, int length, bool align_corners)
{
    float x;
    if (align_corners) {
        x = (n + 1) / 2. * (length - 1);
    } else {
        x = ((n + 1) * length - 1) / 2.;
    }
    return x;
}

static inline float border(float x, float x_min, float x_max)
{
    return UNI_MIN(UNI_MAX(x, x_min), x_max);
}

static inline float reflect(float x, float x_min, float x_max)
{
    float range = x_max - x_min;
    if (x < x_min) {
        float dx = x_min - x;
        int n = dx / range;
        float r = dx - n * range;
        if (n % 2 == 0) {
            x = x_min + r;
        } else {
            x = x_max - r;
        }
    } else if (x > x_max) {
        float dx = x - x_max;
        int n = dx / range;
        float r = dx - n * range;
        if (n % 2 == 0) {
            x = x_max - r;
        } else {
            x = x_min + r;
        }
    }
    return x;
}

template <typename T>
static inline float get(
    const T *image, int it, int ih, int iw, int t, int h, int w, int cAlign, PadMode mode, float *bound)
{
    float pixel;
    if (mode == PAD_CONSTANT) {
        if (t >= 0 && t < it && h >= 0 && h < ih && w >= 0 && w < iw) {
            pixel = image[(((t * ih) + h) * iw + w) * cAlign];
        } else {
            pixel = 0;
        }
    } else if (mode == PAD_EDGE) {
        w = border(w, 0, iw - 1);
        h = border(h, 0, ih - 1);
        //t = border(t, 0, it - 1);
        pixel = image[(((t * ih) + h) * iw + w) * cAlign];
    } else {
        w = reflect(w, bound[0], bound[1]);
        h = reflect(h, bound[2], bound[3]);
        //t = reflect(t, bound[4], bound[5]);
        pixel = image[(((t * ih) + h) * iw + w) * cAlign];
    }
    return pixel;
}

template <typename T>
static EE grid_sample_kernel(TensorDesc inputDesc,
    T *input,
    TensorDesc gridDesc,
    T *grid,
    GridSampleParamSpec p,
    T *tmp,
    TensorDesc outputDesc,
    T *output)
{
    DataType idt;
    DataFormat idf;
    U32 in, ic, it, ih, iw;
    if (tensorIs3d(inputDesc)) {
        CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &in, &ic, &iw));
        it = ih = 1;
    } else if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        it = 1;
    } else if (tensorIs5d(inputDesc)) {
        CHECK_STATUS(tensor5dGet(inputDesc, &idt, &idf, &in, &ic, &it, &ih, &iw));
    } else {
        return NOT_SUPPORTED;
    }
    int olen = tensorNumElements(outputDesc) / in / ic;
    int S = tensorNumElements(gridDesc) / in / olen;
    int cAlign = 1;
    if (idf == DF_NCHWC8) {
        cAlign = 8;
    }
    ic /= cAlign;

    float w_min = -0.5;
    float w_max = iw - 0.5;
    float h_min = -0.5;
    float h_max = ih - 0.5;
    float t_min = -0.5;
    float t_max = it - 0.5;
    if (p.align_corners) {
        w_min = -0.5;
        w_max = iw - 0.5;
        h_min = -0.5;
        h_max = ih - 0.5;
        t_min = -0.5;
        t_max = it - 0.5;
    }
    float bound[6] = {w_min, w_max, h_min, h_max, t_min, t_max};
    EE ret = SUCCESS;
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
    for (U32 o = 0; o < in * ic; o++) {
        U32 n = o / ic;
        U32 c = o % ic;
        float x, y, z;
        for (int i = 0; i < olen; i++) {
            T *g = grid + (n * olen + i) * S;
            for (int c8 = 0; c8 < cAlign; c8++) {
                T *data = input + o * it * ih * iw * cAlign + c8;
                T *out = output + (o * olen + i) * cAlign + c8;
                x = denormalize(g[0], iw, p.align_corners);
                if (S > 1) {
                    y = denormalize(g[1], ih, p.align_corners);
                } else {
                    y = 0;
                }
                if (S > 2) {
                    z = denormalize(g[2], it, p.align_corners);
                } else {
                    z = 0;
                }
                //switch (p.pad_mode) {
                //    case PAD_EDGE: {
                //        x = border(x, 0, iw - 1);
                //        y = border(y, 0, ih - 1);
                //        z = border(z, 0, it - 1);
                //        break;
                //    }
                //    case PAD_REFLECT: {
                //        x = reflect(x, w_min, w_max);
                //        y = reflect(y, h_min, h_max);
                //        z = reflect(z, t_min, t_max);
                //        break;
                //    }
                //    default:
                //        break;
                //}
                switch (p.mode) {
                    case RESIZE_NEAREST: {
                        x = round(x);
                        y = round(y);
                        z = round(z);
                        *out = get(data, it, ih, iw, z, y, x, cAlign, p.pad_mode, bound);
                        break;
                    }
                    case RESIZE_LINEAR: {
                        int x1 = floor(x);
                        int x2 = x1 + 1;
                        int y1 = floor(y);
                        int y2 = y1 + 1;
                        //int z1 = floor(z);
                        //int z2 = z1 + 1;
                        float p11 = get(data, it, ih, iw, 0, y1, x1, cAlign, p.pad_mode, bound);
                        float p12 = get(data, it, ih, iw, 0, y1, x2, cAlign, p.pad_mode, bound);
                        float p21 = get(data, it, ih, iw, 0, y2, x1, cAlign, p.pad_mode, bound);
                        float p22 = get(data, it, ih, iw, 0, y2, x2, cAlign, p.pad_mode, bound);
                        float dx2 = x2 - x;
                        float dx1 = x - x1;
                        float dy2 = y2 - y;
                        float dy1 = y - y1;
                        *out = dy2 * (dx2 * p11 + dx1 * p12) + dy1 * (dx2 * p21 + dx1 * p22);
                        break;
                    }
                    default:
                        UNI_ERROR_LOG("GridSample currently not support this mode.\n");
                        ret = NOT_SUPPORTED;
                        break;
                }
            }
        }
    }
    return ret;
}

EE grid_sample_cpu(TensorDesc inputDesc,
    void *input,
    TensorDesc gridDesc,
    void *grid,
    GridSampleParamSpec p,
    void *tmp,
    TensorDesc outputDesc,
    void *output)
{
    EE ret = NOT_SUPPORTED;
    switch (inputDesc.dt) {
#ifdef _USE_FP16
        case DT_F16: {
            ret = grid_sample_kernel<F16>(inputDesc, (F16 *)input, gridDesc, (F16 *)grid, p,
                (F16 *)tmp, outputDesc, (F16 *)output);
            break;
        }
#endif
#ifdef _USE_FP32
        case DT_F32: {
            ret = grid_sample_kernel<F32>(inputDesc, (F32 *)input, gridDesc, (F32 *)grid, p,
                (F32 *)tmp, outputDesc, (F32 *)output);
            break;
        }
#endif
        default:
            break;
    }
    return ret;
}
