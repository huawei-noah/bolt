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
#if defined(INPUT_TRAN)
mem_trans_nchw_to_ncwhc4_input_tran
#elif defined(OUTPUT_TRAN)
mem_trans_nchw_to_ncwhc4_output_tran
#else
mem_trans_nchw_to_ncwhc4
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
        const __global T *in,
        __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    int ocd4 = (oc + 3) >> 2;
    const int idc = idz % ocd4;
    const int idn = idz / ocd4;
    const int iz_off = (idn * ic + (idc << 2)) * iw_str * ih_str;

#if defined(INPUT_TRAN)
    if (idx >= (oh + 3) >> 2 || idy >= ow) {
        return;
    }
    int in_off = iz_off + (idy + iw_off) * ih_str + (idx << 2) + ih_off + offset_in;
    int out_off = (idz * ow_str + idy + ow_off) * oh_str + (idx << 2) + oh_off;
    char iex = ((idx << 2) + 4 <= ih) ? 4 : (ih & 3);
    char oex = ((idx << 2) + 4 <= oh) ? 4 : (oh & 3);
    if ((idx << 2) >= ih || idy >= iw || idc >= ((ic + 3) >> 2)) {
        iex = 0;
    }
    int out_str = 1;
#else
#if defined(OUTPUT_TRAN)
    if (idx >= (oh + 3) >> 2 || idy >= ow) {
        return;
    }
    int out_off = (idz * ow_str + idy + ow_off) * oh_str + (idx << 2) + oh_off;
    int out_str = 1;
    char oex = ((idx << 2) + 4 <= oh) ? 4 : (oh & 3);
#else
    if (idx >= (ow + 3) >> 2 || idy >= oh) {
        return;
    }
    int out_off = (idz * ow_str + (idx << 2) + ow_off) * oh_str + idy + oh_off;
    int out_str = oh_str;
    char oex = ((idx << 2) + 4 <= ow) ? 4 : (ow & 3);
#endif
    int in_off = iz_off + (idy + ih_off) * iw_str + (idx << 2) + iw_off + offset_in;
    char iex = ((idx << 2) + 4 <= iw) ? 4 : (iw & 3);
    if ((idx << 2) >= iw || idy >= ih || idc >= ((ic + 3) >> 2)) {
        iex = 0;
    }
#endif
    char iec = ((idc << 2) + 4 <= ic) ? 4 : (ic & 3);
    T4 val[4];
    val[0] = 0;
    val[1] = 0;
    val[2] = 0;
    val[3] = 0;

    int iwh_str = iw_str * ih_str;
    if (iex == 4) {
        val[0] = vload4(0, in + in_off);
        if (iec > 1) {
            val[1] = vload4(0, in + in_off + iwh_str);
        }
        if (iec > 2) {
            val[2] = vload4(0, in + in_off + (iwh_str << 1));
        }
        if (iec > 3) {
            val[3] = vload4(0, in + in_off + iwh_str * 3);
        }
    } else {
        if (iex == 1) {
            val[0].x = in[in_off];
            if (iec > 1) {
                val[1].x = in[in_off + iwh_str];
            }
            if (iec > 2) {
                val[2].x = in[in_off + (iwh_str << 1)];
            }
            if (iec > 3) {
                val[3].x = in[in_off + iwh_str * 3];
            }
        }
        if (iex == 2) {
            val[0].xy = vload2(0, in + in_off);
            if (iec > 1) {
                val[1].xy = vload2(0, in + in_off + iwh_str);
            }
            if (iec > 2) {
                val[2].xy = vload2(0, in + in_off + (iwh_str << 1));
            }
            if (iec > 3) {
                val[3].xy = vload2(0, in + in_off + iwh_str * 3);
            }
        }
        if (iex == 3) {
            val[0].xyz = vload3(0, in + in_off);
            if (iec > 1) {
                val[1].xyz = vload3(0, in + in_off + iwh_str);
            }
            if (iec > 2) {
                val[2].xyz = vload3(0, in + in_off + (iwh_str << 1));
            }
            if (iec > 3) {
                val[3].xyz = vload3(0, in + in_off + iwh_str * 3);
            }
        }
    }

    vstore4((T4)(val[0].x, val[1].x, val[2].x, val[3].x), out_off, out + offset_out);
    if (oex > 1) {
        vstore4((T4)(val[0].y, val[1].y, val[2].y, val[3].y), out_off + out_str, out + offset_out);
    }
    if (oex > 2) {
        vstore4((T4)(val[0].z, val[1].z, val[2].z, val[3].z), out_off + (out_str << 1),
            out + offset_out);
    }
    if (oex > 3) {
        vstore4(
            (T4)(val[0].w, val[1].w, val[2].w, val[3].w), out_off + out_str * 3, out + offset_out);
    }
}
