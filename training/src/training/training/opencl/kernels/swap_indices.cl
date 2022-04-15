R"(// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


__kernel void swap_ijk2kji(
	const int dim1,
    const int dim2,
    const int dim3,
    __global const T *src,
    __global T *dst)
{
    int idx = get_global_id(0);
	int idy = get_global_id(1);
    if (idx >= dim1 || idy >= dim2) {
        return;
    }
    
	int dst_off = idx * dim2 * dim3 + idy * dim3;
	int src_off = idy * dim1 + idx;
	int src_stride = dim1 * dim2;
	for (int k = 0; k < dim3; ++k, src_off += src_stride)
	{
		dst[dst_off + k] = src[src_off];
	}
}
)"