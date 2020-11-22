// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#if defined(USE_NCHW)
#else
#endif

#if defined(INPUT_NCHW) && defined(OUTPUT_NCHW)
#define LOAD_VAL(ew, ec, idx, idy, idz, ih_str, iw_str, ih_off, iw_off, buf, val) \
    {                                                                             \
        int off = (idz * ih_str + idy + ih_off) * iw_str + (idx << 2) + iw_off;   \
        val = 0;                                                                  \
        if (ew == 4) {                                                            \
            val = vload4(0, buf + off);                                           \
        } else {                                                                  \
            if (ew == 1) {                                                        \
                val.x = buf[off];                                                 \
            }                                                                     \
            if (ew == 2) {                                                        \
                val.xy = vload2(0, buf + off);                                    \
            }                                                                     \
            if (ew == 3) {                                                        \
                val.xyz = vload3(0, buf + off);                                   \
            }                                                                     \
        }                                                                         \
    }
#define STORE_VAL(ew, ec, idx, idy, idz, oh_str, ow_str, oh_off, ow_off, buf, val) \
    {                                                                              \
        int off = (idz * oh_str + idy + oh_off) * ow_str + (idx << 2) + ow_off;    \
        if (ew == 4) {                                                             \
            vstore4(val, 0, buf + off);                                            \
        } else {                                                                   \
            if (ew == 1) {                                                         \
                buf[off] = val.x;                                                  \
            }                                                                      \
            if (ew == 2) {                                                         \
                vstore2((T2)(val.x, val.y), 0, buf + off);                         \
            }                                                                      \
            if (ew == 3) {                                                         \
                vstore3((T3)(val.x, val.y, val.z), 0, buf + off);                  \
            }                                                                      \
        }                                                                          \
    }
#elif defined(INPUT_NCHW) && defined(OUTPUT_NCWHC4)
#define LOAD_VAL(ew, ec, idx, idy, idz, ih_str, iw_str, ih_off, iw_off, buf, val)      \
    {                                                                                  \
        int off = ((idz << 2) * ih_str + idy + ih_off) * iw_str + (idx << 2) + iw_off; \
        int str = iw_str * ih_str;                                                     \
        if (ew == 4) {                                                                 \
            val[0] = vload4(0, buf + off);                                             \
            if (ec > 1)                                                                \
                val[1] = vload4(0, buf + off + str);                                   \
            if (ec > 2)                                                                \
                val[2] = vload4(0, buf + off + str * 2);                               \
            if (ec > 3)                                                                \
                val[3] = vload4(0, buf + off + str * 3);                               \
        } else {                                                                       \
            if (ew == 1) {                                                             \
                val[0].x = buf[off];                                                   \
                if (ec > 1)                                                            \
                    val[1].x = buf[off + str];                                         \
                if (ec > 2)                                                            \
                    val[2].x = buf[off + str * 2];                                     \
                if (ec > 3)                                                            \
                    val[3].x = buf[off + str * 3];                                     \
            }                                                                          \
            if (ew == 2) {                                                             \
                val[0].xy = vload2(0, buf + off);                                      \
                if (ec > 1)                                                            \
                    val[1].xy = vload2(0, buf + off + str);                            \
                if (ec > 2)                                                            \
                    val[2].xy = vload2(0, buf + off + str * 2);                        \
                if (ec > 3)                                                            \
                    val[3].xy = vload2(0, buf + off + str * 3);                        \
            }                                                                          \
            if (ew == 3) {                                                             \
                val[0].xyz = vload3(0, buf + off);                                     \
                if (ec > 1)                                                            \
                    val[1].xyz = vload3(0, buf + off + str);                           \
                if (ec > 2)                                                            \
                    val[2].xyz = vload3(0, buf + off + str * 2);                       \
                if (ec > 3)                                                            \
                    val[3].xyz = vload3(0, buf + off + str * 3);                       \
            }                                                                          \
        }                                                                              \
    }
#define STORE_VAL(ew, ec, idx, idy, idz, oh_str, ow_str, oh_off, ow_off, buf, val)        \
    {                                                                                     \
        int off = (idz * ow_str + (idx << 2) + ow_off) * oh_str + idy + oh_off;           \
        vstore4((T4)(val[0].x, val[1].x, val[2].x, val[3].x), off, buf);                  \
        if (ew > 1)                                                                       \
            vstore4((T4)(val[0].y, val[1].y, val[2].y, val[3].y), off + oh_str, buf);     \
        if (ew > 2)                                                                       \
            vstore4((T4)(val[0].z, val[1].z, val[2].z, val[3].z), off + oh_str * 2, buf); \
        if (ew > 3)                                                                       \
            vstore4((T4)(val[0].w, val[1].w, val[2].w, val[3].w), off + oh_str * 3, buf); \
    }
#elif defined(INPUT_NCWHC4) && defined(OUTPUT_NCHW)
#define LOAD_VAL(ew, ec, idx, idy, idz, ih_str, iw_str, ih_off, iw_off, buf, val) \
    {                                                                             \
        int off = (idz * iw_str + (idy << 2) + iw_off) * ih_str + idx + ih_off;   \
        val[0] = vload4(off, buf);                                                \
        if (ew > 1)                                                               \
            val[1] = vload4(off + ih_str, buf);                                   \
        if (ew > 2)                                                               \
            val[2] = vload4(off + ih_str * 2, buf);                               \
        if (ew > 3)                                                               \
            val[3] = vload4(off + ih_str * 3, buf);                               \
    }
#define STORE_VAL(ew, ec, idx, idy, idz, oh_str, ow_str, oh_off, ow_off, buf, val)             \
    {                                                                                          \
        int off = ((idz << 2) * oh_str + idx + oh_off) * ow_str + (idy << 2) + ow_off;         \
        int str = ow_str * oh_str;                                                             \
        if (ew == 4) {                                                                         \
            vstore4((T4)(val[0].x, val[1].x, val[2].x, val[3].x), 0, buf + off);               \
            if (ec > 1)                                                                        \
                vstore4((T4)(val[0].y, val[1].y, val[2].y, val[3].y), 0, buf + off + str);     \
            if (ec > 2)                                                                        \
                vstore4((T4)(val[0].z, val[1].z, val[2].z, val[3].z), 0, buf + off + str * 2); \
            if (ec > 3)                                                                        \
                vstore4((T4)(val[0].w, val[1].w, val[2].w, val[3].w), 0, buf + off + str * 3); \
        } else {                                                                               \
            if (ew == 1) {                                                                     \
                buf[off] = val[0].x;                                                           \
                if (ec > 1)                                                                    \
                    buf[off + str] = val[0].y;                                                 \
                if (ec > 2)                                                                    \
                    buf[off + str * 2] = val[0].z;                                             \
                if (ec > 3)                                                                    \
                    buf[off + str * 3] = val[0].w;                                             \
            }                                                                                  \
            if (ew == 2) {                                                                     \
                vstore2((T2)(val[0].x, val[1].x), 0, buf + off);                               \
                if (ec > 1)                                                                    \
                    vstore2((T2)(val[0].y, val[1].y), 0, buf + off + str);                     \
                if (ec > 2)                                                                    \
                    vstore2((T2)(val[0].z, val[1].z), 0, buf + off + str * 2);                 \
                if (ec > 3)                                                                    \
                    vstore2((T2)(val[0].w, val[1].w), 0, buf + off + str * 3);                 \
            }                                                                                  \
            if (ew == 3) {                                                                     \
                vstore3((T3)(val[0].x, val[1].x, val[2].x), 0, buf + off);                     \
                if (ec > 1)                                                                    \
                    vstore3((T3)(val[0].y, val[1].y, val[2].y), 0, buf + off + str);           \
                if (ec > 2)                                                                    \
                    vstore3((T3)(val[0].z, val[1].z, val[2].z), 0, buf + off + str * 2);       \
                if (ec > 3)                                                                    \
                    vstore3((T3)(val[0].w, val[1].w, val[2].w), 0, buf + off + str * 3);       \
            }                                                                                  \
        }                                                                                      \
    }
#else
#define LOAD_VAL(ew, ec, idx, idy, idz, ih_str, iw_str, ih_off, iw_off, buf, val) \
    {                                                                             \
        int off = (idz * iw_str + idy + iw_off) * ih_str + idx + ih_off;          \
        val = vload4(off, buf);                                                   \
    }
#define STORE_VAL(ew, ec, idx, idy, idz, oh_str, ow_str, oh_off, ow_off, buf, val) \
    {                                                                              \
        int off = (idz * ow_str + idy + ow_off) * oh_str + idx + oh_off;           \
        vstore4(val, off, buf);                                                    \
    }
#endif

__kernel void
#if defined(INPUT_NCHW) && defined(OUTPUT_NCHW)
channel_resize_nchw
#elif defined(INPUT_NCHW) && defined(OUTPUT_NCWHC4)
channel_resize_nchw_ncwhc4
#elif defined(INPUT_NCWHC4) && defined(OUTPUT_NCHW)
channel_resize_ncwhc4_nchw
#else
channel_resize
#endif
    (const int ih_str,
        const int iw_str,
        const int ic_str,
        const int ih_off,
        const int iw_off,
        const int oh_str,
        const int ow_str,
        const int oh_off,
        const int ow_off,
        const int in_c,
        const int out_c,
        const int w,
        const int bx,
        const int by,
        const __global const T *in,
        __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }

    char ew = 0;
    char ec = 0;
#if defined(INPUT_NCHW) && defined(OUTPUT_NCHW)
    T4 val = 0;
    ew = ((idx << 2) + 4 <= w) ? 4 : (w & 3);
#elif defined(INPUT_NCHW) && defined(OUTPUT_NCWHC4)
    T4 val[4] = {0};
    ew = ((idx << 2) + 4 <= w) ? 4 : (w & 3);
    ec = ((idz << 2) + 4 <= in_c) ? 4 : (in_c & 3);
#elif defined(INPUT_NCWHC4) && defined(OUTPUT_NCHW)
    T4 val[4] = {0};
    ew = ((idy << 2) + 4 <= w) ? 4 : (w & 3);
    ec = ((idz << 2) + 4 <= out_c) ? 4 : (out_c & 3);
#else
    T4 val = 0;
#endif

    if (idz < ic_str) {
        LOAD_VAL(ew, ec, idx, idy, idz, ih_str, iw_str, ih_off, iw_off, in, val);
    }
    STORE_VAL(ew, ec, idx, idy, idz, oh_str, ow_str, oh_off, ow_off, out, val);
}
