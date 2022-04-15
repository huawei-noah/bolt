R"(// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// bias has size [h]
// data has size [N x h x w], work size is N x h
__kernel void bias_forward(
	const int N,
	const int h,
	const int w,
	__global const T *bias,
	__global T *data)
{
	const int idy = get_global_id(0);
	const int idx = get_global_id(1);

	if (idx >= h || idy >= N)
	{
		return;
	}

	const int offset = idy * w * h;
	const T b = bias[idx];
	for (int i = 0; i < w; ++i)
	{
		data[offset + idx * w + i] += b;
	}
}

#ifdef TF_STYLE
// biasGrad has size [h]
// deltas has size [N x w x h], work size is h
__kernel void bias_backward(
	const int N,
	const int h,
	const int w,
	__global const T *deltas,
	__global T *biasGrad)
{
	const int idx = get_global_id(0);

	if (idx >= h)
	{
		return;
	}

	const int m = N * w;
	int off = idx; 
	T b = 0;
	for (int i = 0; i < m; ++i, off += h)
	{
		b += deltas[off];
	}
	biasGrad[idx] += b;
}

// input has size [iw x h x N]
// output has size [N x h x ow], work size is N x h
__kernel void dilation_forward(
	const int N,
	const int h,
	const int iw,
	const int ow,
	const int is,
	const int os,
	__global const T *input,
	__global T *output)
{
	const int idy = get_global_id(0);
	const int idx = get_global_id(1);

	if (idx >= h || idy >= N)
	{
		return;
	}

	int inOff = idx * N + idy;
	const int outOff = idy * ow * h + idx * ow;
	int off = 0;
	for (int i = 0; i < iw; ++i, off += os, inOff += is)
	{
		output[outOff + off] = input[inOff];
	}
}

// input has size [N x h x iw]
// output has size [ow x h x N], work size is N x h
__kernel void weights_backward(
	const int N,
	const int h,
	const int iw,
	const int ow,
	const int dilation,
	__global const T *input,
	__global T *output)
{
	const int idy = get_global_id(0);
	const int idx = get_global_id(1);

	if (idx >= h || idy >= N)
	{
		return;
	}
	
	int os = h * N;
	int inOff = idy * iw * h + idx * iw;
	int outOff = idy + idx * N;
	for (int i = 0; i < ow; ++i, outOff += os, inOff += dilation)
	{
		output[outOff] += input[inOff];
	}
}
#else

// biasGrad has size [h]
// deltas has size [N x h x w], work size is h
__kernel void bias_backward(
	const int N,
	const int h,
	const int w,
	__global const T *deltas,
	__global T *biasGrad)
{
	const int idx = get_global_id(0);

	if (idx >= h)
	{
		return;
	}

	const int ns = w * h;
	int off = idx * w; 
	T b = 0;
	for (int i = 0; i < N; ++i, off += ns)
	{
		for (int j = 0; j < w; ++j) 
		{
			b += deltas[off + j];
		}
	}
	biasGrad[idx] += b;
}

// input has size [N x h x iw]
// output has size [N x h x ow], work size is N x h
__kernel void dilation_forward(
	const int N,
	const int h,
	const int iw,
	const int ow,
	const int is,
	const int os,
	__global const T *input,
	__global T *output)
{
	const int idy = get_global_id(0);
	const int idx = get_global_id(1);

	if (idx >= h || idy >= N)
	{
		return;
	}

	const int inOff = idy * iw * h + idx * iw;
	const int outOff = idy * ow * h + idx * ow;
	int off = 0;
	for (int i = 0; i < iw; i += is, off += os)
	{
		output[outOff + off] = input[inOff + i];
	}
}



// input has size [N x h x iw]
// output has size [N x h x ow], work size is N x h
__kernel void weights_backward(
	const int N,
	const int h,
	const int iw,
	const int ow,
	const int dilation,
	__global const T *input,
	__global T *output)
{
	const int idy = get_global_id(0);
	const int idx = get_global_id(1);

	if (idx >= h || idy >= N)
	{
		return;
	}

	int inOff = idy * iw * h + idx * iw;
	const int outOff = idy * ow * h + idx * ow;
	int off = 0;
	for (int i = 0; i < ow; ++i, ++off, inOff += dilation)
	{
		output[outOff + off] += input[inOff];
	}
}
#endif
)"