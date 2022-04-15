R"(// Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "kernel_def.h"

__kernel void bn_test_depth(const int batchSize,
	const int depth,
	const int height,
	const int width,
	const T eps,
	__global const T* input,
	__global const T* beta,
	__global const T* gamma,
	__global const T* meanEval,
	__global const T* varEval,
	__global T* output)
{
    const int idx = get_global_id(0);
    const int WH = width * height;
    
    if (idx >= depth) {
        return;
    }
	T meanVal = meanEval[idx];
	T mult = gamma[idx]/ sqrt(varEval[idx] + eps);
	T betaVal = beta[idx];
	
	for (int batch = 0; batch < batchSize; ++batch)
	{
		const int offset = batch * depth * WH + idx * WH;
		for (int i = 0; i < WH; ++i)
		{
			T inputVal = input[offset + i];
			output[offset + i] = mult * (inputVal - meanVal) + betaVal;
		}
	}
}

__kernel void bn_test_width(const int batchSize,
	const int depth,
	const int height,
	const int width,
	const T eps,
	__global const T* input,
	__global const T* beta,
	__global const T* gamma,
	__global const T* meanEval,
	__global const T* varEval,
	__global T* output)
{
    const int idx = get_global_id(0);
    const int len = batchSize * depth * height * width;
	const int stride = width;
	
    if (idx >= width) {
        return;
    }
	T meanVal = meanEval[idx];
	T mult = gamma[idx]/ sqrt(varEval[idx] + eps);
	T betaVal = beta[idx];
	
	for (int off = 0; off < len; off += stride)
	{
		const int offset = off + idx;
		T inputVal = input[offset];
		output[offset] = mult * (inputVal - meanVal) + betaVal;
	}
}

__kernel void bn_forward_depth(const int batchSize,
	const int depth,
	const int height,
	const int width,
	const T reciprocalN,
	const T momentum,
	const T eps,
	const int useMomentum,
	__global const T* input,
	__global const T* beta,
	__global const T* gamma,
    __global T* mean,
	__global T* var,
	__global T* xHat,
	__global T* varSqrt,
	__global T* output,
	__global T* meanEval,
	__global T* varEval)
{
    const int idx = get_global_id(0);
	const int WH = width * height;
    
    if (idx >= depth) {
        return;
    }
	T meanVal = 0;
	for (int batch = 0; batch < batchSize; ++batch)
	{
		const int offset = batch * depth * WH + idx * WH;
		for (int i = 0; i < WH; ++i)
		{
			meanVal += input[offset + i];
		}
	}
	meanVal *= reciprocalN;
	mean[idx] = meanVal;
	T varVal = 0;
	for (int batch = 0; batch < batchSize; ++batch)
	{
		const int offset = batch * depth * WH + idx * WH;
		for (int i = 0; i < WH; ++i)
		{
			T inputVal = input[offset + i];
			varVal += (inputVal-meanVal)*(inputVal-meanVal);
		}
	}
	varVal *= reciprocalN;
	var[idx] = varVal;
	if (useMomentum == 1)
	{
		T curVal = meanEval[idx];
		meanEval[idx] = (1 - momentum) * curVal + momentum * meanVal;
		curVal = varEval[idx];
		varEval[idx] = (1 - momentum) * curVal + momentum * varVal;
	}
	
	const T varSqrtVal = sqrt(varVal + eps);
	varSqrt[idx] = varSqrtVal;
	const T gammaVal = gamma[idx];
	const T betaVal = beta[idx];
	for (int batch = 0; batch < batchSize; ++batch)
	{
		const int offset = batch * depth * WH + idx * WH;
		for (int i = 0; i < WH; ++i)
		{
			const T inputVal = input[offset + i];
			const T xHatVal = (inputVal-meanVal) / varSqrtVal;
			xHat[offset + i] = xHatVal;
			output[offset + i] = gammaVal * xHatVal + betaVal;
		}
	}
}

__kernel void bn_forward_height(const int batchSize,
	const int depth,
	const int height,
	const int width,
	const T reciprocalN,
	const T momentum,
	const T eps,
	const int useMomentum,
	__global const T* input,
	__global const T* beta,
	__global const T* gamma,
    __global T* mean,
	__global T* var,
	__global T* xHat,
	__global T* varSqrt,
	__global T* output,
	__global T* meanEval,
	__global T* varEval)
{
    const int idx = get_global_id(0);
    
    if (idx >= height) {
        return;
    }
	T meanVal = 0;
	for (int batch = 0; batch < batchSize; ++batch)
	{
		for (int c = 0; c < depth; ++c)
		{
			const int offset = batch * depth * width * height + c * width * height + idx * width;
			for (int i = 0; i < width; ++i)
			{
				meanVal += input[offset + i];
			}
		}
	}
	meanVal *= reciprocalN;
	mean[idx] = meanVal;
	T varVal = 0;
	for (int batch = 0; batch < batchSize; ++batch)
	{
		for (int c = 0; c < depth; ++c)
		{
			const int offset = batch * depth * width * height + c * width * height + idx * width;
			for (int i = 0; i < width; ++i)
		{
			T inputVal = input[offset + i];
			varVal += (inputVal-meanVal)*(inputVal-meanVal);
		}}
	}
	varVal *= reciprocalN;
	var[idx] = varVal;
	if (useMomentum == 1)
	{
		T curVal = meanEval[idx];
		meanEval[idx] = (1 - momentum) * curVal + momentum * meanVal;
		curVal = varEval[idx];
		varEval[idx] = (1 - momentum) * curVal + momentum * varVal;
	}
	
	const T varSqrtVal = sqrt(varVal + eps);
	varSqrt[idx] = varSqrtVal;
	const T gammaVal = gamma[idx];
	const T betaVal = beta[idx];
	for (int batch = 0; batch < batchSize; ++batch)
	{
		for (int c = 0; c < depth; ++c)
		{
			const int offset = batch * depth * width * height + c * width * height + idx * width;
			for (int i = 0; i < width; ++i)
		{
			const T inputVal = input[offset + i];
			const T xHatVal = (inputVal-meanVal) / varSqrtVal;
			xHat[offset + i] = xHatVal;
			output[offset + i] = gammaVal * xHatVal + betaVal;
		}}
	}
}

__kernel void bn_forward_width(const int batchSize,
	const int depth,
	const int height,
	const int width,
	const T reciprocalN,
	const T momentum,
	const T eps,
	const int useMomentum,
	__global const T* input,
	__global const T* beta,
	__global const T* gamma,
    __global T* mean,
	__global T* var,
	__global T* xHat,
	__global T* varSqrt,
	__global T* output,
	__global T* meanEval,
	__global T* varEval)
{
    const int idx = get_global_id(0);
    const int len = batchSize * depth * height * width;
	const int stride = width;
    if (idx >= width) {
        return;
    }
	T meanVal = 0;
	for (int off = 0; off < len; off += stride)
	{
		meanVal += input[off + idx];
	}
	meanVal *= reciprocalN;
	mean[idx] = meanVal;
	T varVal = 0;
	for (int off = 0; off < len; off += stride)
	{
		const int offset = off + idx;
		T inputVal = input[offset];
		varVal += (inputVal-meanVal)*(inputVal-meanVal);
	}
	
	varVal *= reciprocalN;
	var[idx] = varVal;
	if (useMomentum == 1)
	{
		T curVal = meanEval[idx];
		meanEval[idx] = (1 - momentum) * curVal + momentum * meanVal;
		curVal = varEval[idx];
		varEval[idx] = (1 - momentum) * curVal + momentum * varVal;
	}
	
	const T varSqrtVal = sqrt(varVal + eps);
	varSqrt[idx] = varSqrtVal;
	const T gammaVal = gamma[idx];
	const T betaVal = beta[idx];
	
	for (int off = 0; off < len; off += stride)
	{
		const int offset = off + idx;
		const T inputVal = input[offset];
		const T xHatVal = (inputVal-meanVal) / varSqrtVal;
		xHat[offset] = xHatVal;
		output[offset] = gammaVal * xHatVal + betaVal;
	}
}

__kernel void bn_backward_depth(const int batchSize,
	const int depth,
	const int height,
	const int width,
	__global const T* deltas,
	__global const T* xHat,
	__global const T* varSqrt,
	__global const T* gamma,
	__global T* prevDeltas,
	__global T* nablaBeta,
	__global T* nablaGamma)
{
	const int idx = get_global_id(0);
    const int WH = width * height;
	
    if (idx >= depth) {
        return;
    }
	
	for (int batch = 0; batch < batchSize; ++batch)
	{
		const int offset = batch * depth * WH + idx * WH;
		for (int i = 0; i < WH; ++i)
		{
			const T deltaVal = deltas[offset + i];
			const T xHatVal = xHat[offset + i];
			nablaBeta[idx] += deltaVal;
			nablaGamma[idx] += deltaVal * xHatVal;
		}
	}
	
	T dvar = 0;
	T dvar2 = 0;
	const T gammaVal = gamma[idx];
	for (int batch = 0; batch < batchSize; ++batch)
	{
		const int offset = batch * depth * WH + idx * WH;
		for (int i = 0; i < WH; ++i)
		{
			const T deltaVal = deltas[offset + i];
			const T xHatVal = xHat[offset + i];
			const T nablaXHatVal = deltaVal * gammaVal;
			dvar += nablaXHatVal;
			dvar2 += nablaXHatVal * xHatVal;
		}
	}
	const T N = (T)(batchSize * WH);
	const T varSqrtVal = varSqrt[idx];
	const T divisor = (T)1 / (N * varSqrtVal);
	for (int batch = 0; batch < batchSize; ++batch)
	{
		const int offset = batch * depth * WH + idx * WH;
		for (int i = 0; i < WH; ++i)
		{
			const T deltaVal = deltas[offset + i];
			const T xHatVal = xHat[offset + i];
			const T nablaXHatVal = deltaVal * gammaVal;
			T prevDeltaVal = (N * nablaXHatVal - dvar - xHatVal * dvar2) * divisor ;
			prevDeltas[offset + i] += prevDeltaVal;
		}
	}
}

__kernel void bn_backward_width(const int batchSize,
	const int depth,
	const int height,
	const int width,
	__global const T* deltas,
	__global const T* xHat,
	__global const T* varSqrt,
	__global const T* gamma,
	__global T* prevDeltas,
	__global T* nablaBeta,
	__global T* nablaGamma)
{
	const int idx = get_global_id(0);
	const int len = batchSize * depth * height * width;
	const int stride = width;
	
    if (idx >= width) {
        return;
    }
	
	for (int off = 0; off < len; off += stride)
	{
		const int offset = off + idx;
		const T deltaVal = deltas[offset];
		const T xHatVal = xHat[offset];
		nablaBeta[idx] += deltaVal;
		nablaGamma[idx] += deltaVal * xHatVal;
	}
	
	T dvar = 0;
	T dvar2 = 0;
	const T gammaVal = gamma[idx];
	for (int off = 0; off < len; off += stride)
	{
		const int offset = off + idx;
		const T deltaVal = deltas[offset];
		const T xHatVal = xHat[offset];
		const T nablaXHatVal = deltaVal * gammaVal;
		dvar += nablaXHatVal;
		dvar2 += nablaXHatVal * xHatVal;
	}
	
	const T N = (T)(batchSize * depth * height);
	const T varSqrtVal = varSqrt[idx];
	const T divisor = (T)1 / (N * varSqrtVal);
	for (int off = 0; off < len; off += stride)
	{
		const int offset = off + idx;
		const T deltaVal = deltas[offset];
		const T xHatVal = xHat[offset];
		const T nablaXHatVal = deltaVal * gammaVal;
		T prevDeltaVal = (N * nablaXHatVal - dvar - xHatVal * dvar2) * divisor ;
		prevDeltas[offset] += prevDeltaVal;
	}
}

__kernel void bn_backward_depth_frozen(const int batchSize,
	const int depth,
	const int height,
	const int width,
	__global const T* deltas,
	__global const T* xHat,
	__global const T* varSqrt,
	__global const T* gamma,
	__global T* prevDeltas)
{
	const int idx = get_global_id(0);
    const int WH = width * height;
	
    if (idx >= depth) {
        return;
    }
	
	
	T dvar = 0;
	T dvar2 = 0;
	const T gammaVal = gamma[idx];
	for (int batch = 0; batch < batchSize; ++batch)
	{
		const int offset = batch * depth * WH + idx * WH;
		for (int i = 0; i < WH; ++i)
		{
			const T deltaVal = deltas[offset + i];
			const T xHatVal = xHat[offset + i];
			const T nablaXHatVal = deltaVal * gammaVal;
			dvar += nablaXHatVal;
			dvar2 += nablaXHatVal * xHatVal;
		}
	}
	const T N = (T)(batchSize * WH);
	const T varSqrtVal = varSqrt[idx];
	const T divisor = (T)1 / (N * varSqrtVal);
	for (int batch = 0; batch < batchSize; ++batch)
	{
		const int offset = batch * depth * WH + idx * WH;
		for (int i = 0; i < WH; ++i)
		{
			const T deltaVal = deltas[offset + i];
			const T xHatVal = xHat[offset + i];
			const T nablaXHatVal = deltaVal * gammaVal;
			T prevDeltaVal = (N * nablaXHatVal - dvar - xHatVal * dvar2) * divisor ;
			prevDeltas[offset + i] += prevDeltaVal;
		}
	}
}

__kernel void bn_backward_width_frozen(const int batchSize,
	const int depth,
	const int height,
	const int width,
	__global const T* deltas,
	__global const T* xHat,
	__global const T* varSqrt,
	__global const T* gamma,
	__global T* prevDeltas)
{
	const int idx = get_global_id(0);
	const int len = batchSize * depth * height * width;
	const int stride = width;
	
    if (idx >= width) {
        return;
    }
	
	
	T dvar = 0;
	T dvar2 = 0;
	const T gammaVal = gamma[idx];
	for (int off = 0; off < len; off += stride)
	{
		const int offset = off + idx;
		const T deltaVal = deltas[offset];
		const T xHatVal = xHat[offset];
		const T nablaXHatVal = deltaVal * gammaVal;
		dvar += nablaXHatVal;
		dvar2 += nablaXHatVal * xHatVal;
	}

	const T N = (T)(batchSize * depth * height);
	const T varSqrtVal = varSqrt[idx];
	const T divisor = (T)1 / (N * varSqrtVal);
	for (int off = 0; off < len; off += stride)
	{
		const int offset = off + idx;
		const T deltaVal = deltas[offset];
		const T xHatVal = xHat[offset];
		const T nablaXHatVal = deltaVal * gammaVal;
		T prevDeltaVal = (N * nablaXHatVal - dvar - xHatVal * dvar2) * divisor ;
		prevDeltas[offset] += prevDeltaVal;
	}
}

)"