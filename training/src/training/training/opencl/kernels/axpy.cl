R"(// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#if defined(USE_HALF)
#define READ_IMAGE(image, sampler, coord) read_imageh(image, sampler, coord)
#define WRITE_IMAGE(image, coord, data) write_imageh(image, coord, data)
#else
#define READ_IMAGE(image, sampler, coord) read_imagef(image, sampler, coord)
#define WRITE_IMAGE(image, coord, data) write_imagef(image, coord, data)
#endif
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void axpy(
    const int len,
	const T alpha,
    const int x_off0,
	const int y_off0,
	const int bx,
    __global const T *x,
    __global T *y)
{
    const int idx = get_global_id(0);
    if (idx >= bx) {
        return;
    }
	const int idx4 = idx << 2;
	const int x_off = x_off0 + idx4;
	const int y_off = y_off0 + idx4;
    char ew = ((idx4 + 4) <= len) ? 4 : (len & 3);
    T4 sum = 0;
    T4 val = 0;
    
    if (ew == 4) {
        val = vload4(0, y + y_off);
        sum = vload4(0, x + x_off);
    } else {
        if (ew == 1) {
            val.x = y[y_off];
            sum.x = x[x_off];
        }
        if (ew == 2) {
            val.xy = vload2(0, y + y_off);
            sum.xy = vload2(0, x + x_off);
        }
        if (ew == 3) {
            val.xyz = vload3(0, y + y_off);
            sum.xyz = vload3(0, x + x_off);
        }
    }
    val += alpha * sum;
    
    if (ew == 4)
    {
        vstore4(val, 0, y + y_off);
    }
    else
    {
        if (ew == 1)
        {
            y[y_off] = val.x;
        }
        if (ew == 2)
        {
            vstore2(val.xy, 0, y + y_off);
        }
        if (ew == 3)
        {
            vstore3(val.xyz, 0, y+ y_off);
        }
    }
}
)"