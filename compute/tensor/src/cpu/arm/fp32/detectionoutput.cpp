// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/fp32/tensor_computing_fp32.h"

EE decode_priorbox_fp32(const F32 *location,
    const F32 *priorbox,
    const F32 *variance,
    I32 num_total_priorbox,
    F32 *xmin,
    F32 *ymin,
    F32 *xmax,
    F32 *ymax)
{
    I32 i = 0;
    for (; i < num_total_priorbox - 3; i += 4) {
        float32x4x4_t loc = vld4q_f32(location + i * 4);
        float32x4x4_t pb = vld4q_f32(priorbox + i * 4);
        float32x4x4_t var = vld4q_f32(variance + i * 4);

        float32x4_t pb_w = vsubq_f32(pb.val[2], pb.val[0]);
        float32x4_t pb_h = vsubq_f32(pb.val[3], pb.val[1]);
        float32x4_t pb_cx = vmulq_n_f32(vaddq_f32(pb.val[0], pb.val[2]), 0.5f);
        float32x4_t pb_cy = vmulq_n_f32(vaddq_f32(pb.val[1], pb.val[3]), 0.5f);

        float32x4_t box_cx = vfmaq_f32(pb_cx, var.val[0], vmulq_f32(loc.val[0], pb_w));
        float32x4_t box_cy = vfmaq_f32(pb_cy, var.val[1], vmulq_f32(loc.val[1], pb_h));
        float32x4_t box_w =
            vmulq_f32(vexpq_f32_03_percent_error(vmulq_f32(var.val[2], loc.val[2])), pb_w);
        float32x4_t box_h =
            vmulq_f32(vexpq_f32_03_percent_error(vmulq_f32(var.val[3], loc.val[3])), pb_h);

        vst1q_f32(xmin + i, vfmaq_n_f32(box_cx, box_w, -0.5f));
        vst1q_f32(ymin + i, vfmaq_n_f32(box_cy, box_h, -0.5f));
        vst1q_f32(xmax + i, vfmaq_n_f32(box_cx, box_w, 0.5f));
        vst1q_f32(ymax + i, vfmaq_n_f32(box_cy, box_h, 0.5f));
    }
    for (; i < num_total_priorbox; i++) {
        const F32 *loc = location + i * 4;
        const F32 *pb = priorbox + i * 4;
        const F32 *var = variance + i * 4;

        F32 pb_w = pb[2] - pb[0];
        F32 pb_h = pb[3] - pb[1];
        F32 pb_cx = (pb[0] + pb[2]) * 0.5f;
        F32 pb_cy = (pb[1] + pb[3]) * 0.5f;

        F32 box_cx = var[0] * loc[0] * pb_w + pb_cx;
        F32 box_cy = var[1] * loc[1] * pb_h + pb_cy;
        F32 box_w = static_cast<F32>(exp(var[2] * loc[2]) * pb_w);
        F32 box_h = static_cast<F32>(exp(var[3] * loc[3]) * pb_h);

        xmin[i] = box_cx + box_w * -0.5f;
        ymin[i] = box_cy + box_h * -0.5f;
        xmax[i] = box_cx + box_w * 0.5f;
        ymax[i] = box_cy + box_h * 0.5f;
    }
    return SUCCESS;
}
