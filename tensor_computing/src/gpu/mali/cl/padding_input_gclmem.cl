// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.





__kernel void padding_input_gclmem(const int iw, const int ih, const int pw, const int ph, 
                                   const int ow, const int oh, const __global const T* in, __global T* out){

    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);

    int bx = iw >> 2;
    int rx = iw & 3;
    T4 val = 0;

    int in_off  = (idz * ih + idy) * iw + (idx << 2);
    int out_off = (idz * oh + idy + ph) * ow + (idx << 2) + pw;
    if(rx == 0 || idx < bx){
        val = vload4(0, in + in_off);
    } else {
        if(rx == 1)  val.x = in[in_off];
        if(rx == 2) {val.x = in[in_off]; val.y = in[in_off + 1];}    
        if(rx == 3) {val.x = in[in_off]; val.y = in[in_off + 1]; val.z = in[in_off + 2];}    
    }
    vstore4(val, 0, out + out_off);
}

