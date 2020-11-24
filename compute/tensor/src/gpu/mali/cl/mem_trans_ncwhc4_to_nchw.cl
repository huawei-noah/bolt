// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void
#if defined(OUTPUT_TRAN)
mem_trans_ncwhc4_to_nchw_output_tran
#else
mem_trans_ncwhc4_to_nchw
#endif
    (const int iw_str,
        const int ih_str,
        const int iw_off,
        const int ih_off,
        const int ow_str,
        const int oh_str,
        const int ow_off,
        const int oh_off,
        const int iw,
        const int ih,
        const int ic,
        const int ow,
        const int oh,
        const int oc,
        const int offset_in,
        const int offset_out,
        __global T *in,
        __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
#if defined(OUTPUT_TRAN)
    if (idx >= (ow + 3) >> 2 || idy >= oh) {
        return;
    }
    int in_off = (idz * iw_str + idy + iw_off) * ih_str + (idx << 2) + ih_off;
    int out_off = ((idz << 2) * oh_str + idy + oh_off) * ow_str + (idx << 2) + ow_off + offset_out;
    char iex = ((idx << 2) + 4 <= ih) ? 4 : (ih & 3);
    char oex = ((idx << 2) + 4 <= ow) ? 4 : (ow & 3);
    if ((idx << 2) >= ih || idy >= iw || idz >= (ic + 3) >> 2) {
        iex = 0;
    }
    const int in_str = 1;
#else
    if (idx >= oh || idy >= (ow + 3) >> 2) {
        return;
    }
    int in_off = (idz * iw_str + (idy << 2) + iw_off) * ih_str + idx + ih_off;
    int out_off = ((idz << 2) * oh_str + idx + oh_off) * ow_str + (idy << 2) + ow_off + offset_out;
    char iex = ((idy << 2) + 4 <= iw) ? 4 : (iw & 3);
    char oex = ((idy << 2) + 4 <= ow) ? 4 : (ow & 3);
    if (idx >= ih || (idy << 2) >= iw || idz >= (ic + 3) >> 2) {
        iex = 0;
    }
    const int in_str = ih_str;
#endif
    short oec = ((idz << 2) + 4 <= oc) ? 4 : (oc & 3);
    T4 val[4];
    val[0] = 0;
    val[1] = 0;
    val[2] = 0;
    val[3] = 0;

    if (iex > 0) {
        val[0] = vload4(in_off, in + offset_in);
    }
    if (iex > 1) {
        val[1] = vload4(in_off + in_str, in + offset_in);
    }
    if (iex > 2) {
        val[2] = vload4(in_off + (in_str << 1), in + offset_in);
    }
    if (iex > 3) {
        val[3] = vload4(in_off + in_str * 3, in + offset_in);
    }

    int owh_str = ow_str * oh_str;
    if (oex == 4) {
        vstore4((T4)(val[0].x, val[1].x, val[2].x, val[3].x), 0, out + out_off);
        if (oec > 1) {
            vstore4((T4)(val[0].y, val[1].y, val[2].y, val[3].y), 0, out + out_off + owh_str);
        }
        if (oec > 2) {
            vstore4((T4)(val[0].z, val[1].z, val[2].z, val[3].z), 0, out + out_off + (owh_str << 1));
        }
        if (oec > 3) {
            vstore4((T4)(val[0].w, val[1].w, val[2].w, val[3].w), 0, out + out_off + owh_str * 3);
        }
    } else {
        if (oex == 1) {
            out[out_off] = val[0].x;
            if (oec > 1) {
                out[out_off + owh_str] = val[0].y;
            }
            if (oec > 2) {
                out[out_off + (owh_str << 1)] = val[0].z;
            }
            if (oec > 3) {
                out[out_off + owh_str * 3] = val[0].w;
            }
        }
        if (oex == 2) {
            vstore2((T2)(val[0].x, val[1].x), 0, out + out_off);
            if (oec > 1) {
                vstore2((T2)(val[0].y, val[1].y), 0, out + out_off + owh_str);
            }
            if (oec > 2) {
                vstore2((T2)(val[0].z, val[1].z), 0, out + out_off + (owh_str << 1));
            }
            if (oec > 3) {
                vstore2((T2)(val[0].w, val[1].w), 0, out + out_off + owh_str * 3);
            }
        }
        if (oex == 3) {
            vstore3((T3)(val[0].x, val[1].x, val[2].x), 0, out + out_off);
            if (oec > 1) {
                vstore3((T3)(val[0].y, val[1].y, val[2].y), 0, out + out_off + owh_str);
            }
            if (oec > 2) {
                vstore3((T3)(val[0].z, val[1].z, val[2].z), 0, out + out_off + (owh_str << 1));
            }
            if (oec > 3) {
                vstore3((T3)(val[0].w, val[1].w, val[2].w), 0, out + out_off + owh_str * 3);
            }
        }
    }
}
