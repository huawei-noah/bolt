// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.





__kernel void normalization(const int len, const int on, const int ih_str, const int ic_str, const int ih_off, const int iw_off, const int oh_str, const int oh_off, const int ow_off, 
    __global const T* alpha, __global const T* beta, __global const T* in, __global T* out) {
    int idx = get_global_id(0);
    if(idx >= len) return;

    float mean = 0;
    float var  = 0;
    float para = 1.0 / on;
  
    int in_off = iw_off * ih_str + idx + ih_off;
    for(int i = 0; i < ic_str; ++i) {
        T4 tmp = vload4(in_off + i * ih_str, in);
        float4 tmpf;
        tmpf.x = tmp.x;
        tmpf.y = tmp.y;
        tmpf.z = tmp.z;
        tmpf.w = tmp.w;
        mean += (float)(tmpf.x + tmpf.y + tmpf.z + tmpf.w);
    }
    mean = mean * para;

    for(int i = 0; i < ic_str; ++i) {
        T4 tmp = vload4(in_off + i * ih_str, in);
        float4 tmpf;
        tmpf.x = tmp.x;
        tmpf.y = tmp.y;
        tmpf.z = tmp.z;
        tmpf.w = tmp.w;
        tmpf.x = tmpf.x - mean;
        tmpf.y = tmpf.y - mean;
        tmpf.z = tmpf.z - mean;
        tmpf.w = tmpf.w - mean;
        var += tmpf.x * tmpf.x + tmpf.y * tmpf.y + tmpf.z * tmpf.z + tmpf.w * tmpf.w;
    }
    var = var * para;

    float std_val = sqrt(var + 1e-6);
    std_val = 1.0 / std_val;
    int out_off = ow_off * oh_str + idx + oh_off;
    for(int i = 0; i < ic_str; ++i) {
        T4 out_val = vload4(in_off + i * ih_str, in);
        T4 alp     = vload4(i, alpha);
        T4 bet     = vload4(i, beta);
        out_val.x = alp.x * (out_val.x - mean) * std_val + bet.x;
        out_val.y = alp.y * (out_val.y - mean) * std_val + bet.y;
        out_val.z = alp.z * (out_val.z - mean) * std_val + bet.z;
        out_val.w = alp.w * (out_val.w - mean) * std_val + bet.w;
        vstore4(out_val, out_off + i * oh_str, out);
    }
}
