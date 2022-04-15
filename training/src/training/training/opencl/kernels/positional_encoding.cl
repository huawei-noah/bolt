R"(// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



__kernel void durations_range(
	const int batch_size,
	const int iw,
    const int ow,
    __global T *input,
    __global T *output)
{
    int id = get_global_id(0);
	if (id >= batch_size)
	{
		return;
	}
	const int i_off = id * iw;
	const int o_off = id * ow;
	int index = 0;
	for (int i = 0; (i < iw && index < ow); ++i)
	{
		int el = (int)(input[i_off + i]);
		for (int j = 0; (j < el && index < ow); ++j)
		{
			output[o_off + index++] = j;
		}
	}
}


__kernel void range_lut(
	const int batch_size,
	const int iw,
	const int ow,
	const int lut_h,
	const int lut_w,
    __global const T *input,
    __global const T *lut,
    __global T *output)
{
    int idy = get_global_id(0);
    int idx = get_global_id(1);
	if (idy >= batch_size || idx >= iw)
	{
		return;
	}
	const int index = (int)input[idy * iw + idx];
	if (index >= lut_h)
	{
		return;
	}
	const int lut_off = index * lut_w;
	const int o_off = idy * iw * ow + idx * ow;
	for (int i = 0; i < ow; ++i)
	{
		//printf("%d, %d: %d - %f", idy, idx, o_off + i, lut[lut_off + i]);
		output[o_off + i] = lut[lut_off + i];
	}
}
)"