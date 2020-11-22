// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void deconv_direct(__global const T *input,
    __global const T *weights,
    __global T *output,
    __read_only image1d_t bias,
    int iw,
    int iw_str,
    int iw_off,
    int ih,
    int ih_str,
    int ih_off,
    int kw,
    int kh,
    int kc,
    int kn,
    int sw,
    int sh,
    int pw,
    int ph,
    int ow,
    int ow_str,
    int ow_off,
    int oh,
    int oh_str,
    int oh_off,
    int ic,
    int oc,
    int align_h,
    int align_w,
    int in_channel_blocks,
    int out_channel_blocks)
{
    const int oh_idx = get_global_id(0);
    const int ow_idx = get_global_id(1);
    const int oc_idx = get_global_id(2);
    if (oh_idx >= oh || ow_idx >= ow || oc_idx >= oc) {
        return;
    }

    T4 out0 = read_imageh(bias, sampler, oc_idx);

    int kernel_start_x = max(0, (oh_idx + align_h) / sh);
    int kernel_start_y = max(0, (ow_idx + align_w) / sw);

    int deal_kernel_width = kw - (kernel_start_y * sw + pw) + ow_idx - 1;
    int deal_kernel_height = kh - (kernel_start_x * sh + ph) + oh_idx - 1;

    int kernel_0, kernel_1, kernel_2, kernel_3, kernel_y;
    T4 in0;
    T4 weights0, weights1, weights2, weights3;
    int in_off, kernel_off;
    for (int i = 0; i < in_channel_blocks; i++) {
        kernel_0 = 0;
        kernel_1 = kernel_0 + 1;
        kernel_2 = kernel_0 + 2;
        kernel_3 = kernel_0 + 3;
        for (int k_y = deal_kernel_width, idx_w = kernel_start_y; k_y >= 0; k_y -= sw, idx_w++) {
            int in_width0 = idx_w;
            int in_height0 = kernel_start_x;
            for (int k_x = deal_kernel_height; k_x >= 0; k_x -= sh) {
                kernel_off =
                    (oc_idx * kw * kh * in_channel_blocks + i * kw * kh + k_x * kh + k_y) * 4;
                weights0 = vload4(kernel_off + kernel_0, weights);
                weights1 = vload4(kernel_off + kernel_1, weights);
                weights2 = vload4(kernel_off + kernel_2, weights);
                weights3 = vload4(kernel_off + kernel_3, weights);

                // in_off = i * ih * iw + ih * in_width0 + in_height0;
                in_off = (i * iw_str + in_width0 + iw_off) * ih_str + ih_off + in_height0;
                if (in_height0 < 0 || in_height0 >= ih || in_width0 < 0 || in_width0 >= iw) {
                    in0 = (T4)0;
                } else {
                    in0 = vload4(in_off, input);
                }

                out0 = mad(in0.x, weights0, out0);
                out0 = mad(in0.y, weights1, out0);
                out0 = mad(in0.z, weights2, out0);
                out0 = mad(in0.w, weights3, out0);
                in_height0++;
            }
        }
    }
    int out_off = (oc_idx * ow_str + ow_idx + ow_off) * oh_str + oh_idx + oh_off;
    vstore4(out0, out_off, output);
}
