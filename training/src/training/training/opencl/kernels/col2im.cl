R"(// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void col2im(const int iw,
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
    const int c_im = get_global_id(0);
	const int h_pad = get_global_id(1);
	const int w_pad = get_global_id(2);
	
	const int wc = (iw + 2 * pw - kw) / sw + 1;
    const int hc = (ih + 2 * ph - kh) / sh + 1;
	
	const int cb = c_im * kw * kh;
	const int ce = (c_im + 1) * kw * kh;
	
	T val = 0;
	for (int c = cb; c < ce; ++c)
	{
		const int w_offset = c % kw;
		const int h_offset = (c / kw) % kh;
		
		if ((w_pad - w_offset + pw) % sw != 0)
		{
			continue;
		}
		if ((h_pad - h_offset + ph) % sh != 0)
		{
			continue;
		}
		
		const int w = (w_pad - w_offset + pw) / sw;
		const int h = (h_pad - h_offset + ph) / sh;
		
		if (h < 0 || h >= hc || w < 0 || w >= wc)
		{
			continue;
		}
		
		val += in[(c * hc + h) * wc + w + in_offset];
	}
	out[(c_im * ih + h_pad) * iw + w_pad + out_offset] += val;
}

__kernel void col2im_reversed(const int iw,
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
    const int c_im = get_global_id(0);
	const int h_pad = get_global_id(1);
	const int w_pad = get_global_id(2);
	
	
	
	const int wc = (iw + 2 * pw - kw) / sw + 1;
    const int hc = (ih + 2 * ph - kh) / sh + 1;
	
	const int cb = c_im * kw * kh;
	const int ce = (c_im + 1) * kw * kh;
	
	T val = 0;
	
	for (int c = cb; c < ce; ++c)
	{
		const int w_offset = c % kw;
		const int h_offset = (c / kw) % kh;
		
		if ((w_pad - w_offset + pw) % sw != 0)
		{
			continue;
		}
		if ((h_pad - h_offset + ph) % sh != 0)
		{
			continue;
		}
		
		const int w = (w_pad - w_offset + pw) / sw;
		const int h = (h_pad - h_offset + ph) / sh;
		
		
		
		
		if (h < 0 || h >= hc || w < 0 || w >= wc)
		{
			continue;
		}
		
		
		val += in[(c * hc + h) * wc + w + in_offset];
	}
	out[(w_pad * ih + h_pad) * ic + c_im + out_offset] += val;
}
)"