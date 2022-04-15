R"(// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
__kernel void hswishBackward(const int size,
    const T a,
    const T b,
    const T c,
    __global T *input,
    __global T *deltas,
    __global T *prevLayerDelta)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx == 0 && idy == 0 && idz == 0)
    {
        for (int q = 0; q < size; ++q)
        {
            if (input[q] > 3.0f)
            {
                prevLayerDelta[q] += deltas[q];
            }
            else if (input[q] > -3.0f && input[q] < 3.0f)
            {
                prevLayerDelta[q] += deltas[q] * (input[q] / 3.0f + 0.5f);
            }
            else if (input[q] == 3.0f)
            {
                prevLayerDelta[q] += a * deltas[q] / 2.0f + b * deltas[q] * (input[q] / 3.0f + 0.5f) / 2.0f;
            }
            else if (input[q] == -3.0f)
            {
                prevLayerDelta[q] += c * deltas[q] * (input[q] / 3.0f + 0.5f) / 2.0f;
            }
        }
    }
};
)"
