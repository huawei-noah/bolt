R"(// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void im2col(const int iw,
    const int ih,
    const int ic,
    const int kw,
    const int kh,
    const int pw,
    const int ph,
    const int sw,
    const int sh,
    __global const T *in,
	const int in_offset,
    __global T *out,
	const int out_offset)
{
    const int c = get_global_id(0);
	const int h = get_global_id(1);
	const int w = get_global_id(2);
	
	const int w_offset = c % kw;
	const int h_offset = (c / kw) % kh;
	const int c_im = c / (kh * kw);
	
	const int wc = (iw + 2 * pw - kw) / sw + 1;
    const int hc = (ih + 2 * ph - kh) / sh + 1;	
	
	const int w_pad = w * sw - pw + w_offset;
	const int h_pad = h * sh - ph + h_offset;
	
	if (h_pad >= 0 && h_pad < ih && w_pad >= 0 && w_pad < iw)
	{
	    out[(c * hc + h) * wc + w + out_offset] = in[(c_im * ih + h_pad) * iw + w_pad + in_offset];
	}
	else
	{
	    out[(c * hc + h) * wc + w + out_offset] = 0;
	}
}

__kernel void im2col_reversed(const int iw,
    const int ih,
    const int ic,
    const int kw,
    const int kh,
    const int pw,
    const int ph,
    const int sw,
    const int sh,
    __global const T *in,
	const int in_offset,
    __global T *out,
	const int out_offset)
{
    const int c = get_global_id(0);
	const int h = get_global_id(1);
	const int w = get_global_id(2);
	
	const int w_offset = c % kw;
	const int h_offset = (c / kw) % kh;
	const int c_im = c / (kh * kw);
	
	const int wc = (iw + 2 * pw - kw) / sw + 1;
    const int hc = (ih + 2 * ph - kh) / sh + 1;	
	
	const int w_pad = w * sw - pw + w_offset;
	const int h_pad = h * sh - ph + h_offset;
	
	if (h_pad >= 0 && h_pad < ih && w_pad >= 0 && w_pad < iw)
	{
	    out[(c * hc + h) * wc + w + out_offset] = in[(w_pad * ih + h_pad) * ic + c_im + in_offset];
	}
	else
	{
	    out[(c * hc + h) * wc + w + out_offset] = 0;
	}
}
)"