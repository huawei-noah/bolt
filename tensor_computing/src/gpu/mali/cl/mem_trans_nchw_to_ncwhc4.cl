// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.





__kernel void mem_trans_nchw_to_ncwhc4(const int iw, const int ih, const int ic, const int iwh_str, const int pw, const int ph, 
                                       const int ow, const int oh, const __global const T* in, __global T* out){

    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);

    short ew = ((idx << 2) + 4 <= iw) ? 4 : (iw & 3);
    short ec = ((idz << 2) + 4 <= ic) ? 4 : (ic & 3);
    T4 val[4];
    val[0] = 0;
    val[1] = 0;
    val[2] = 0;
    val[3] = 0;
   
    int in_off  = ((idz << 2) * ih + idy) * iw + (idx << 2);

    if(ew == 4){
                   val[0] = vload4(0, in + in_off);
        if(ec > 1) val[1] = vload4(0, in + in_off + iwh_str);
        if(ec > 2) val[2] = vload4(0, in + in_off +(iwh_str << 1));
        if(ec > 3) val[3] = vload4(0, in + in_off + iwh_str * 3);
    } else {
        if(ew == 1){
                       val[0].x = in[in_off];
            if(ec > 1) val[1].x = in[in_off + iwh_str];
            if(ec > 2) val[2].x = in[in_off +(iwh_str << 1)];
            if(ec > 3) val[3].x = in[in_off + iwh_str * 3];
        }
        if(ew == 2){
                       val[0].xy = vload2(0, in + in_off);
            if(ec > 1) val[1].xy = vload2(0, in + in_off + iwh_str);
            if(ec > 2) val[2].xy = vload2(0, in + in_off +(iwh_str << 1));
            if(ec > 3) val[3].xy = vload2(0, in + in_off + iwh_str * 3);
        }
        if(ew == 3){
                       val[0].xyz = vload3(0, in + in_off);
            if(ec > 1) val[1].xyz = vload3(0, in + in_off + iwh_str);
            if(ec > 2) val[2].xyz = vload3(0, in + in_off +(iwh_str << 1));
            if(ec > 3) val[3].xyz = vload3(0, in + in_off + iwh_str * 3);
        }
    }

    int out_off = (idz * ow + (idx << 2) + pw) * oh + idy + ph;
               vstore4((T4)(val[0].x, val[1].x, val[2].x, val[3].x), out_off,            out);
    if(ew > 1) vstore4((T4)(val[0].y, val[1].y, val[2].y, val[3].y), out_off + oh,       out);
    if(ew > 2) vstore4((T4)(val[0].z, val[1].z, val[2].z, val[3].z), out_off +(oh << 1), out);
    if(ew > 3) vstore4((T4)(val[0].w, val[1].w, val[2].w, val[3].w), out_off + oh * 3,   out);
}

