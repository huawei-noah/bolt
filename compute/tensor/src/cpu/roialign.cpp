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

template <typename T>
static F32 bilinear_interpolate(T *data, U32 w, U32 h, F32 x, F32 y)
{
    if (y < -1.0 || y > h || x < -1.0 || x > w) {
        return 0;
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

    F32 hx = x1 - x;
    F32 lx = x - x0;
    F32 hy = y1 - y;
    F32 ly = y - y0;

    if (x1 >= w) {
        x1 = w - 1;
        hx = 1.f;
        lx = 0.f;
    }
    if (y1 >= h) {
        y1 = h - 1;
        hy = 1.f;
        ly = 0.f;
    }

    F32 r0 = data[y0 * w + x0] * hx + data[y0 * w + x1] * lx;
    F32 r1 = data[y1 * w + x0] * hx + data[y1 * w + x1] * lx;

    F32 val = r0 * hy + r1 * ly;
    return val;
}

template <typename T>
static EE roialign_kernel(std::vector<void *> input,
    T *output,
    std::vector<TensorDesc> inputDesc,
    U32 output_h,
    U32 output_w,
    U32 sampling_ratio,
    F32 spatial_scale)
{
    DataType idt0, idt1;
    DataFormat idf0, idf1;
    U32 in0, ic0, ih0, iw0;
    U32 ih1, iw1;
    CHECK_STATUS(tensor4dGet(inputDesc[0], &idt0, &idf0, &in0, &ic0, &ih0, &iw0));
    CHECK_STATUS(tensor2dGet(inputDesc[1], &idt1, &idf1, &ih1, &iw1));
    T *feature_map = (T *)input[0];
    T *rois = (T *)input[1];
    CHECK_REQUIREMENT(idf0 == DF_NCHWC8 || idf0 == DF_NCHW);
    if (inputDesc[0].df == DF_NCHWC8) {
        T *tmp = (T *)malloc(tensorNumBytes(inputDesc[0]));
        memcpy(tmp, feature_map, tensorNumBytes(inputDesc[0]));
        CHECK_STATUS(transformToNCHW(inputDesc[0], tmp, inputDesc[0], feature_map));
        free(tmp);
    }

    U32 channel = ic0;
    U32 feature_w = iw0;
    U32 feature_h = ih0;
    U32 num_rois = ih1;
    for (U32 n = 0; n < num_rois; n++) {
        U32 idx_n = n * channel * output_w * output_h;
        F32 roi_start_x1 = static_cast<F32>(rois[n * 4]) * spatial_scale;
        F32 roi_start_y1 = static_cast<F32>(rois[n * 4 + 1]) * spatial_scale;
        F32 roi_end_x2 = static_cast<F32>(rois[n * 4 + 2]) * spatial_scale;
        F32 roi_end_y2 = static_cast<F32>(rois[n * 4 + 3]) * spatial_scale;

        F32 roi_w = std::max(roi_end_x2 - roi_start_x1, 1.f);
        F32 roi_h = std::max(roi_end_y2 - roi_start_y1, 1.f);

        F32 bin_size_w = roi_w / static_cast<F32>(output_w);
        F32 bin_size_h = roi_h / static_cast<F32>(output_h);

        U32 bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_w / output_w);
        U32 bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_h / output_h);

        F32 count = bin_grid_h * bin_grid_w;
        for (U32 c = 0; c < channel; c++) {
            U32 idx_nc = idx_n + c * output_h * output_w;
            T *feature_map_offset = feature_map + c * feature_h * feature_w;
            for (U32 ph = 0; ph < output_h; ph++) {
                for (U32 pw = 0; pw < output_w; pw++) {
                    U32 idx = idx_nc + ph * output_w + pw;
                    F32 output_val = 0;
                    F32 start_x = roi_start_x1 + pw * bin_size_w;
                    F32 start_y = roi_start_y1 + ph * bin_size_h;
                    start_x = std::min(std::max(start_x, 0.f), (F32)feature_w);
                    start_y = std::min(std::max(start_y, 0.f), (F32)feature_h);
                    for (U32 by = 0; by < bin_grid_h; by++) {
                        F32 y = start_y +
                            static_cast<F32>(by + 0.5f) * bin_size_h / static_cast<F32>(bin_grid_h);
                        for (U32 bx = 0; bx < bin_grid_w; bx++) {
                            F32 x = start_x +
                                static_cast<F32>(bx + 0.5f) * bin_size_w /
                                    static_cast<F32>(bin_grid_w);
                            F32 val = bilinear_interpolate<T>(
                                (T *)feature_map_offset, feature_w, feature_h, x, y);
                            output_val += val;
                        }
                    }
                    output_val /= count;
                    output[idx] = output_val;
                }
            }
        }
    }

    return SUCCESS;
}

EE roialign_cpu(std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    RoiAlignParamSpec roiAlignParamSpec,
    TensorDesc outputDesc,
    void *output)
{
    UNUSED(outputDesc);
    if (nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    U32 output_h = roiAlignParamSpec.output_h;
    U32 output_w = roiAlignParamSpec.output_w;
    U32 sampling_ratio = roiAlignParamSpec.sampling_ratio;
    F32 spatial_scale = roiAlignParamSpec.spatial_scale;
    EE ret = SUCCESS;
    switch (inputDesc[0].dt) {
#ifdef _USE_FP32
        case DT_F32:
            ret = roialign_kernel<F32>(
                input, (F32 *)output, inputDesc, output_h, output_w, sampling_ratio, spatial_scale);
            break;
#endif
#ifdef _USE_FP16
        case DT_F16:
            ret = roialign_kernel<F16>(
                input, (F16 *)output, inputDesc, output_h, output_w, sampling_ratio, spatial_scale);
            break;
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
