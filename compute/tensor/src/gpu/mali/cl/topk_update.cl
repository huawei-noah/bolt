// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#if defined(USE_MAX)
#define UP_VAL -65535
#elif defined(USE_MIN)
#define UP_VAL 65535
#endif

#define SWAP_UP(x, y, ix, iy) \
    {                         \
        if (x < y) {          \
            T z = y;          \
            y = x;            \
            x = z;            \
            ushort iz = iy;   \
            iy = ix;          \
            ix = iz;          \
        }                     \
    }

#define SWAP_DN(x, y, ix, iy) \
    {                         \
        if (x > y) {          \
            T z = y;          \
            y = x;            \
            x = z;            \
            uchar iz = iy;    \
            iy = ix;          \
            ix = iz;          \
        }                     \
    }

#define SORT_VAL(v, id)                    \
    {                                      \
        SWAP_DN(v.s0, v.s1, id.s0, id.s1); \
        SWAP_UP(v.s2, v.s3, id.s2, id.s3); \
        SWAP_DN(v.s4, v.s5, id.s4, id.s5); \
        SWAP_UP(v.s6, v.s7, id.s6, id.s7); \
        SWAP_DN(v.s8, v.s9, id.s8, id.s9); \
        SWAP_UP(v.sa, v.sb, id.sa, id.sb); \
        SWAP_DN(v.sc, v.sd, id.sc, id.sd); \
        SWAP_UP(v.se, v.sf, id.se, id.sf); \
                                           \
        \    
    SWAP_DN(v.s0, v.s2, id.s0, id.s2);     \
        SWAP_DN(v.s1, v.s3, id.s1, id.s3); \
        SWAP_UP(v.s4, v.s6, id.s4, id.s6); \
        SWAP_UP(v.s5, v.s7, id.s5, id.s7); \
        SWAP_DN(v.s8, v.sa, id.s8, id.sa); \
        SWAP_DN(v.s9, v.sb, id.s9, id.sb); \
        SWAP_UP(v.sc, v.se, id.sc, id.se); \
        SWAP_UP(v.sd, v.sf, id.sd, id.sf); \
                                           \
        SWAP_DN(v.s0, v.s1, id.s0, id.s1); \
        SWAP_DN(v.s2, v.s3, id.s2, id.s3); \
        SWAP_UP(v.s4, v.s5, id.s4, id.s5); \
        SWAP_UP(v.s6, v.s7, id.s6, id.s7); \
        SWAP_DN(v.s8, v.s9, id.s8, id.s9); \
        SWAP_DN(v.sa, v.sb, id.sa, id.sb); \
        SWAP_UP(v.sc, v.sd, id.sc, id.sd); \
        SWAP_UP(v.se, v.sf, id.se, id.sf); \
                                           \
        SWAP_DN(v.s0, v.s4, id.s0, id.s4); \
        SWAP_DN(v.s1, v.s5, id.s1, id.s5); \
        SWAP_DN(v.s2, v.s6, id.s2, id.s6); \
        SWAP_DN(v.s3, v.s7, id.s3, id.s7); \
        SWAP_UP(v.s8, v.sc, id.s8, id.sc); \
        SWAP_UP(v.s9, v.sd, id.s9, id.sd); \
        SWAP_UP(v.sa, v.se, id.sa, id.se); \
        SWAP_UP(v.sb, v.sf, id.sb, id.sf); \
                                           \
        SWAP_DN(v.s0, v.s2, id.s0, id.s2); \
        SWAP_DN(v.s1, v.s3, id.s1, id.s3); \
        SWAP_DN(v.s4, v.s6, id.s4, id.s6); \
        SWAP_DN(v.s5, v.s7, id.s5, id.s7); \
        SWAP_UP(v.s8, v.sa, id.s8, id.sa); \
        SWAP_UP(v.s9, v.sb, id.s9, id.sb); \
        SWAP_UP(v.sc, v.se, id.sc, id.se); \
        SWAP_UP(v.sd, v.sf, id.sd, id.sf); \
                                           \
        SWAP_DN(v.s0, v.s1, id.s0, id.s1); \
        SWAP_DN(v.s2, v.s3, id.s2, id.s3); \
        SWAP_DN(v.s4, v.s5, id.s4, id.s5); \
        SWAP_DN(v.s6, v.s7, id.s6, id.s7); \
        SWAP_UP(v.s8, v.s9, id.s8, id.s9); \
        SWAP_UP(v.sa, v.sb, id.sa, id.sb); \
        SWAP_UP(v.sc, v.sd, id.sc, id.sd); \
        SWAP_UP(v.se, v.sf, id.se, id.sf); \
                                           \
        SWAP_DN(v.s0, v.s8, id.s0, id.s8); \
        SWAP_DN(v.s1, v.s9, id.s1, id.s9); \
        SWAP_DN(v.s2, v.sa, id.s2, id.sa); \
        SWAP_DN(v.s3, v.sb, id.s3, id.sb); \
        SWAP_DN(v.s4, v.sc, id.s4, id.sc); \
        SWAP_DN(v.s5, v.sd, id.s5, id.sd); \
        SWAP_DN(v.s6, v.se, id.s6, id.se); \
        SWAP_DN(v.s7, v.sf, id.s7, id.sf); \
                                           \
        SWAP_DN(v.s0, v.s4, id.s0, id.s4); \
        SWAP_DN(v.s1, v.s5, id.s1, id.s5); \
        SWAP_DN(v.s2, v.s6, id.s2, id.s6); \
        SWAP_DN(v.s3, v.s7, id.s3, id.s7); \
        SWAP_DN(v.s8, v.sc, id.s8, id.sc); \
        SWAP_DN(v.s9, v.sd, id.s9, id.sd); \
        SWAP_DN(v.sa, v.se, id.sa, id.se); \
        SWAP_DN(v.sb, v.sf, id.sb, id.sf); \
                                           \
        SWAP_DN(v.s0, v.s2, id.s0, id.s2); \
        SWAP_DN(v.s1, v.s3, id.s1, id.s3); \
        SWAP_DN(v.s4, v.s6, id.s4, id.s6); \
        SWAP_DN(v.s5, v.s7, id.s5, id.s7); \
        SWAP_DN(v.s8, v.sa, id.s8, id.sa); \
        SWAP_DN(v.s9, v.sb, id.s9, id.sb); \
        SWAP_DN(v.sc, v.se, id.sc, id.se); \
        SWAP_DN(v.sd, v.sf, id.sd, id.sf); \
                                           \
        SWAP_DN(v.s0, v.s1, id.s0, id.s1); \
        SWAP_DN(v.s2, v.s3, id.s2, id.s3); \
        SWAP_DN(v.s4, v.s5, id.s4, id.s5); \
        SWAP_DN(v.s6, v.s7, id.s6, id.s7); \
        SWAP_DN(v.s8, v.s9, id.s8, id.s9); \
        SWAP_DN(v.sa, v.sb, id.sa, id.sb); \
        SWAP_DN(v.sc, v.sd, id.sc, id.sd); \
        SWAP_DN(v.se, v.sf, id.se, id.sf); \
    }

#define UPDATE_VAL(id, warp, i, buf)      \
    {                                     \
        if (id == i.sf) {                 \
            buf[warp * 16 + 15] = UP_VAL; \
        } else if (id == i.se) {          \
            buf[warp * 16 + 14] = UP_VAL; \
        } else if (id == i.sd) {          \
            buf[warp * 16 + 13] = UP_VAL; \
        } else if (id == i.sc) {          \
            buf[warp * 16 + 12] = UP_VAL; \
        } else if (id == i.sb) {          \
            buf[warp * 16 + 11] = UP_VAL; \
        } else if (id == i.sa) {          \
            buf[warp * 16 + 10] = UP_VAL; \
        } else if (id == i.s9) {          \
            buf[warp * 16 + 9] = UP_VAL;  \
        } else if (id == i.s8) {          \
            buf[warp * 16 + 8] = UP_VAL;  \
        } else if (id == i.s7) {          \
            buf[warp * 16 + 7] = UP_VAL;  \
        } else if (id == i.s6) {          \
            buf[warp * 16 + 6] = UP_VAL;  \
        } else if (id == i.s5) {          \
            buf[warp * 16 + 5] = UP_VAL;  \
        } else if (id == i.s4) {          \
            buf[warp * 16 + 4] = UP_VAL;  \
        } else if (id == i.s3) {          \
            buf[warp * 16 + 3] = UP_VAL;  \
        } else if (id == i.s2) {          \
            buf[warp * 16 + 2] = UP_VAL;  \
        } else if (id == i.s1) {          \
            buf[warp * 16 + 1] = UP_VAL;  \
        } else if (id == i.s0) {          \
            buf[warp * 16] = UP_VAL;      \
        }                                 \
    }

__kernel void
#if defined(USE_MAX)
topk_update_max
#elif defined(USE_MIN)
topk_update_min
#endif
    (const int need_out_id,
        const int out_id_off,
        const int out_id_num,
        const int bx,
        __global const ushort *buf_remove_idx,
        __global T *buf,
        __global ushort *buf_id,
        __global int *out_id)
{
    const int idx = get_global_id(0);
    if (idx >= bx) {
        return;
    }
    ushort id = buf_remove_idx[idx];
    ushort id_warp = id >> 4;
    ushort16 res_id = vload16(id_warp, buf_id);
    UPDATE_VAL(id, id_warp, res_id, buf);
    barrier(CLK_GLOBAL_MEM_FENCE);
    T16 res = vload16(id_warp, buf);
    barrier(CLK_GLOBAL_MEM_FENCE);
    SORT_VAL(res, res_id);
    vstore16(res, id_warp, buf);
    vstore16(res_id, id_warp, buf_id);

    if (need_out_id) {
        int idx_tran = idx;
#if defined(USE_MAX)
        if (out_id_num < 16) {
            idx_tran = 15 - idx;
        }
#endif
        if (idx_tran < out_id_num) {
            out_id[out_id_off + idx_tran] = id;
        }
    }
}
