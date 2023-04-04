// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/fp16/tensor_computing_fp16.h"

EE decode_priorbox_fp16(const F16 *location,
    const F16 *priorbox,
    const F16 *variance,
    I32 num_total_priorbox,
    F16 *xmin,
    F16 *ymin,
    F16 *xmax,
    F16 *ymax)
{
    I32 i = 0;
    for (; i < num_total_priorbox - 7; i += 8) {
        float16x8x4_t loc = vld4q_f16(location + i * 4);
        float16x8x4_t pb = vld4q_f16(priorbox + i * 4);
        float16x8x4_t var = vld4q_f16(variance + i * 4);

        float16x8_t pb_w = vsubq_f16(pb.val[2], pb.val[0]);
        float16x8_t pb_h = vsubq_f16(pb.val[3], pb.val[1]);
        float16x8_t pb_cx = vmulq_n_f16(vaddq_f16(pb.val[0], pb.val[2]), 0.5f);
        float16x8_t pb_cy = vmulq_n_f16(vaddq_f16(pb.val[1], pb.val[3]), 0.5f);

        float16x8_t box_cx = vfmaq_f16(pb_cx, var.val[0], vmulq_f16(loc.val[0], pb_w));
        float16x8_t box_cy = vfmaq_f16(pb_cy, var.val[1], vmulq_f16(loc.val[1], pb_h));
        float16x8_t box_w = vmulq_f16(vexpq_f16_f32(vmulq_f16(var.val[2], loc.val[2])), pb_w);
        float16x8_t box_h = vmulq_f16(vexpq_f16_f32(vmulq_f16(var.val[3], loc.val[3])), pb_h);

        vst1q_f16(xmin + i, vfmaq_n_f16(box_cx, box_w, -0.5f));
        vst1q_f16(ymin + i, vfmaq_n_f16(box_cy, box_h, -0.5f));
        vst1q_f16(xmax + i, vfmaq_n_f16(box_cx, box_w, 0.5f));
        vst1q_f16(ymax + i, vfmaq_n_f16(box_cy, box_h, 0.5f));
    }
    for (; i < num_total_priorbox; i++) {
        const F16 *loc = location + i * 4;
        const F16 *pb = priorbox + i * 4;
        const F16 *var = variance + i * 4;

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
