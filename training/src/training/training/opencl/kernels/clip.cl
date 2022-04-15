R"(// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define LOAD_VAL(ew, idx, idy, idz, ih_str, iw_str, ih_off, iw_off, buf, val)   \
    {                                                                           \
        int off = (idz * ih_str + idy + ih_off) * iw_str + (idx << 2) + iw_off; \
        val = 0;                                                                \
        if (ew == 4) {                                                          \
            val = vload4(0, buf + off);                                         \
        } else {                                                                \
            if (ew == 1)                                                        \
                val.x = buf[off];                                               \
            if (ew == 2) {                                                      \
                T2 tmp = vload2(0, buf + off);                                  \
                val.x = tmp.x;                                                  \
                val.y = tmp.y;                                                  \
            }                                                                   \
            if (ew == 3) {                                                      \
                T3 tmp = vload3(0, buf + off);                                  \
                val.x = tmp.x;                                                  \
                val.y = tmp.y;                                                  \
                val.z = tmp.z;                                                  \
            }                                                                   \
        }                                                                       \
    }
#define STORE_VAL(ew, idx, idy, idz, oh_str, ow_str, oh_off, ow_off, buf, val)  \
    {                                                                           \
        int off = (idz * oh_str + idy + oh_off) * ow_str + (idx << 2) + ow_off; \
        if (ew == 4) {                                                          \
            vstore4(val, 0, buf + off);                                         \
        } else {                                                                \
            if (ew == 1)                                                        \
                buf[off] = val.x;                                               \
            if (ew == 2) {                                                      \
                vstore2((T2)(val.x, val.y), 0, buf + off);                      \
            }                                                                   \
            if (ew == 3) {                                                      \
                vstore3((T3)(val.x, val.y, val.z), 0, buf + off);               \
            }                                                                   \
        }                                                                       \
    }

__kernel void clipForward(const int w,
    const int h,
    const int ih_str,
    const int iw_str,
    const int ih_off,
    const int iw_off,
    const int oh_str,
    const int ow_str,
    const int oh_off,
    const int ow_off,
    const float min_value,
    const float max_value,
    __global T *input,
    __global T *output)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx >= w || idy >= h) {
        return;
    }

    T4 val;
    char ew = 0;
    ew = ((idx << 2) + 4 <= iw_str) ? 4 : (iw_str & 3);
    float4 val_f32;
    LOAD_VAL(ew, idx, idy, idz, ih_str, iw_str, ih_off, iw_off, input, val);
    val_f32.s0 = (float)val.s0;
    val_f32.s1 = (float)val.s1;
    val_f32.s2 = (float)val.s2;
    val_f32.s3 = (float)val.s3;
    val_f32.s0 = clamp(val_f32.s0, min_value, max_value);
    val_f32.s1 = clamp(val_f32.s1, min_value, max_value);
    val_f32.s2 = clamp(val_f32.s2, min_value, max_value);
    val_f32.s3 = clamp(val_f32.s3, min_value, max_value);
    val.s0 = (T)val_f32.s0;
    val.s1 = (T)val_f32.s1;
    val.s2 = (T)val_f32.s2;
    val.s3 = (T)val_f32.s3;

    STORE_VAL(ew, idx, idy, idz, oh_str, ow_str, oh_off, ow_off, output, val);
}

__kernel void clipBackward(const int w,
    const int h,
    const int ih_str,
    const int iw_str,
    const int ih_off,
    const int iw_off,
    const int oh_str,
    const int ow_str,
    const int oh_off,
    const int ow_off,
    const float min_value,
    const float max_value,
    __global T *input,
    __global T *deltas,
    __global T *prevLayerDelta)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx >= w || idy >= h) {
        return;
    }

    T4 val;
    T4 del;
    T4 prev;
    char ew = 0;
    ew = ((idx << 2) + 4 <= iw_str) ? 4 : (iw_str & 3);
    LOAD_VAL(ew, idx, idy, idz, ih_str, iw_str, ih_off, iw_off, input, val);
    LOAD_VAL(ew, idx, idy, idz, ih_str, iw_str, ih_off, iw_off, deltas, del);
    float4 val_f32;
    val_f32.s0 = (float)val.s0;
    val_f32.s1 = (float)val.s1;
    val_f32.s2 = (float)val.s2;
    val_f32.s3 = (float)val.s3;
    val_f32.s0 = (val_f32.s0 >= min_value) * (val_f32.s0 <= max_value);
    val_f32.s1 = (val_f32.s1 >= min_value) * (val_f32.s1 <= max_value);
    val_f32.s2 = (val_f32.s2 >= min_value) * (val_f32.s2 <= max_value);
    val_f32.s3 = (val_f32.s3 >= min_value) * (val_f32.s3 <= max_value);
    val.s0 = (T)val_f32.s0;
    val.s1 = (T)val_f32.s1;
    val.s2 = (T)val_f32.s2;
    val.s3 = (T)val_f32.s3;
    val.s0 *= del.s0;
    val.s1 *= del.s1;
    val.s2 *= del.s2;
    val.s3 *= del.s3;

    LOAD_VAL(ew, idx, idy, idz, ih_str, iw_str, ih_off, iw_off, prevLayerDelta, prev);
    val.s0 += prev.s0;
    val.s1 += prev.s1;
    val.s2 += prev.s2;
    val.s3 += prev.s3;

    STORE_VAL(ew, idx, idy, idz, oh_str, ow_str, oh_off, ow_off, prevLayerDelta, val);
}
)"