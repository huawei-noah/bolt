R"(// Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void gradient_clipping(
    const int len,
	const int bx,
    const T clipNorm,
    __global const T *currGlobalNorm,
    __global T *grad)
{
    const int idx = get_global_id(0);
    if (idx >= bx) {
        return;
    }
    
    if (currGlobalNorm[0] == 0.0f) {
        return;
    }

    const T factor = clipNorm / max(sqrt(currGlobalNorm[0]), clipNorm);
    if (factor == 1.0f) {
        return;
    }

	const int off = idx << 2;
    char ew = ((off + 4) <= len) ? 4 : (len & 3);
    T4 val = 0;
    
    if (ew == 4) {
        val = vload4(0, grad + off);
    } else {
        if (ew == 1) {
            val.x = grad[off];
        }
        if (ew == 2) {
            val.xy = vload2(0, grad + off);
        }
        if (ew == 3) {
            val.xyz = vload3(0, grad + off);
        }
    }
    
    val = factor * val;
    
    if (ew == 4)
    {
        vstore4(val, 0, grad + off);
    }
    else
    {
        if (ew == 1)
        {
            grad[off] = val.x;
        }
        if (ew == 2)
        {
            vstore2(val.xy, 0, grad + off);
        }
        if (ew == 3)
        {
            vstore3(val.xyz, 0, grad + off);
        }
    }
}
)"