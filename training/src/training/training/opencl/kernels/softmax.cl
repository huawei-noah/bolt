R"(// Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (theSoftware"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDEDAS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void softmaxForward(const int externalDimSize,
    const int internalDimSize,
    __global T *input,
    __global T *output)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx == 0 && idy == 0 && idz == 0)
    {
        for (int q = 0; q < externalDimSize; ++q)
        {
            T sum = 0.0f;
            for (int i = 0; i < internalDimSize; ++i)
            {
                output[q * internalDimSize + i] = exp(input[q * internalDimSize + i]);
                sum += output[q * internalDimSize + i];
            }
            for (int i = 0; i < internalDimSize; ++i)
            {
                output[q * internalDimSize + i] /= sum;
            }
        }
    }
}

__kernel void softmaxBackward(const int externalDimSize,
    const int internalDimSize,
    __global T *output,
    __global T *deltas,
    __global T *prevLayerDelta)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx == 0 && idy == 0 && idz == 0)
    {
        for (int q = 0; q < externalDimSize; ++q)
        {
            for (int i = 0; i < internalDimSize; ++i)
            {
                T sum = 0.0f;
                for (int j = 0; j < internalDimSize; ++j)
                {
                    sum += deltas[q * internalDimSize + j] * ((j == i) ? output[q * internalDimSize + j] * (1.0f - output[q * internalDimSize + j]) : -output[q * internalDimSize + i] * output[q * internalDimSize + j]);
                }
                prevLayerDelta[q * internalDimSize + i] += sum;
            }
        }
    }
}

)"