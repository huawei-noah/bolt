// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/tensor_computing_cpu.h"
#include "tensor_transpose.h"

static void preprocess(U32 w, U32 h, F32 x, F32 y, int c8Align, F32 *factor, U32 *offset)
{
    if (y < -1.0 || y > h || x < -1.0 || x > w) {
        UNI_MEMSET(factor, 0, sizeof(float) * 4);
        UNI_MEMSET(offset, 0, sizeof(U32) * 4);
        return;
    }
    if (y <= 0) {
        y = 0;
    }
    if (x <= 0) {
        x = 0;
    }

    U32 x0 = x;
    U32 x1 = x0 + 1;
    U32 y0 = y;
    U32 y1 = y0 + 1;

    if (y0 >= h - 1) {
        y0 = y1 = h - 1;
        y = y0;
    }
    if (x0 >= w - 1) {
        x0 = x1 = w - 1;
        x = x0;
    }
    F32 lx = x - x0;
    F32 ly = y - y0;
    F32 hx = 1 - lx;
    F32 hy = 1 - ly;
    factor[0] = hy * hx;
    factor[1] = hy * lx;
    factor[2] = ly * hx;
    factor[3] = ly * lx;
    offset[0] = (y0 * w + x0) * c8Align;
    offset[1] = (y0 * w + x1) * c8Align;
    offset[2] = (y1 * w + x0) * c8Align;
    offset[3] = (y1 * w + x1) * c8Align;
}

template <typename T, PoolingMode mode>
static void roialign_kernel(std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    U32 output_h,
    U32 output_w,
    U32 sampling_ratio,
    F32 spatial_scale,
    T *output)
{
    DataType idt0, idt1;
    DataFormat idf0, idf1;
    U32 in0, ic0, ih0, iw0;
    U32 ih1, iw1;
    CHECK_STATUS(tensor4dGet(inputDesc[0], &idt0, &idf0, &in0, &ic0, &ih0, &iw0));
    CHECK_STATUS(tensor2dGet(inputDesc[1], &idt1, &idf1, &ih1, &iw1));
    T *feature_map = (T *)input[0];
    T *rois = (T *)input[1];
    U32 c8Align = 1;
    if (idf0 == DF_NCHWC8) {
        c8Align = 8;
    }

    U32 channel = ic0;
    U32 feature_w = iw0;
    U32 feature_h = ih0;
    U32 num_rois = ih1;
    F32 val;
    for (U32 n = 0, idx = 0; n < num_rois; n++) {
        F32 roi_start_x1 = static_cast<F32>(rois[n * 4]) * spatial_scale;
        F32 roi_start_y1 = static_cast<F32>(rois[n * 4 + 1]) * spatial_scale;
        F32 roi_end_x2 = static_cast<F32>(rois[n * 4 + 2]) * spatial_scale;
        F32 roi_end_y2 = static_cast<F32>(rois[n * 4 + 3]) * spatial_scale;

        F32 roi_w = UNI_MAX(roi_end_x2 - roi_start_x1, 1.f);
        F32 roi_h = UNI_MAX(roi_end_y2 - roi_start_y1, 1.f);

        F32 bin_size_w = roi_w / static_cast<F32>(output_w);
        F32 bin_size_h = roi_h / static_cast<F32>(output_h);

        U32 bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_w / output_w);
        U32 bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_h / output_h);

        std::vector<F32> factor(output_h * output_w * bin_grid_h * bin_grid_w * 4);
        std::vector<U32> offset(output_h * output_w * bin_grid_h * bin_grid_w * 4);
        for (U32 ph = 0, id = 0; ph < output_h; ph++) {
            F32 start_y = roi_start_y1 + ph * bin_size_h;
            for (U32 pw = 0; pw < output_w; pw++) {
                F32 start_x = roi_start_x1 + pw * bin_size_w;
                for (U32 by = 0; by < bin_grid_h; by++) {
                    F32 y = start_y +
                        static_cast<F32>(by + 0.5f) * bin_size_h / static_cast<F32>(bin_grid_h);
                    for (U32 bx = 0; bx < bin_grid_w; bx++, id += 4) {
                        F32 x = start_x +
                            static_cast<F32>(bx + 0.5f) * bin_size_w / static_cast<F32>(bin_grid_w);
                        preprocess(feature_w, feature_h, x, y, c8Align, factor.data() + id,
                            offset.data() + id);
                    }
                }
            }
        }
        F32 count = bin_grid_h * bin_grid_w;
        for (U32 c0 = 0, c = 0; c0 < channel / c8Align; c0++) {
            for (U32 c1 = 0; c1 < c8Align; c1++, c++) {
                T *data = feature_map + c0 * feature_h * feature_w * c8Align + c1;
                for (U32 ph = 0, id00 = 0; ph < output_h; ph++) {
                    for (U32 pw = 0; pw < output_w; pw++, idx++) {
                        if (mode == POOLING_MEAN) {
                            val = 0;
                        } else {
                            val = -UNI_F16_MAX;
                        }
                        for (U32 by = 0; by < bin_grid_h; by++) {
                            for (U32 bx = 0; bx < bin_grid_w; bx++, id00 += 4) {
                                int id01 = id00 + 1;
                                int id10 = id00 + 2;
                                int id11 = id00 + 3;
                                if (mode == POOLING_MEAN) {
                                    val += factor[id00] * data[offset[id00]] +
                                        factor[id01] * data[offset[id01]] +
                                        factor[id10] * data[offset[id10]] +
                                        factor[id11] * data[offset[id11]];
                                } else {
                                    val = UNI_MAX(
                                        UNI_MAX(
                                            UNI_MAX(UNI_MAX(val, factor[id00] * data[offset[id00]]),
                                                factor[id01] * data[offset[id01]]),
                                            factor[id10] * data[offset[id10]]),
                                        factor[id11] * data[offset[id11]]);
                                }
                            }
                        }
                        output[idx] = val;
                        if (mode == POOLING_MEAN) {
                            output[idx] /= count;
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
static EE roialign_kernel(std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    PoolingMode mode,
    U32 output_h,
    U32 output_w,
    U32 sampling_ratio,
    F32 spatial_scale,
    T *output)
{
    EE ret = SUCCESS;
    switch (mode) {
        case POOLING_MEAN: {
            roialign_kernel<T, POOLING_MEAN>(
                inputDesc, input, output_h, output_w, sampling_ratio, spatial_scale, output);
            break;
        }
        case POOLING_MAX: {
            roialign_kernel<T, POOLING_MAX>(
                inputDesc, input, output_h, output_w, sampling_ratio, spatial_scale, output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE roialign_cpu(std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    RoIAlignParamSpec p,
    TensorDesc outputDesc,
    void *output)
{
    UNUSED(outputDesc);
    if (nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    EE ret = NOT_SUPPORTED;
    switch (inputDesc[0].dt) {
#ifdef _USE_FP32
        case DT_F32:
            ret = roialign_kernel<F32>(inputDesc, input, p.mode, p.output_h, p.output_w,
                p.sampling_ratio, p.spatial_scale, (F32 *)output);
            break;
#endif
#ifdef _USE_FP16
        case DT_F16:
            ret = roialign_kernel<F16>(inputDesc, input, p.mode, p.output_h, p.output_w,
                p.sampling_ratio, p.spatial_scale, (F16 *)output);
            break;
#endif
        default:
            break;
    }
    return ret;
}
