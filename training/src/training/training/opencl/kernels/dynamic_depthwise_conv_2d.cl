R"(// Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define MANGLE_NAME_IMPL(base, PATH, OUTPUT) base##PATH##OUTPUT
#define MANGLE_NAME(base, PATH, OUTPUT) MANGLE_NAME_IMPL(base, PATH, OUTPUT)

__kernel void MANGLE_NAME(dynamicDepthwiseConv2D, PATH, OUTPUT)(const int batchSize,
    const int inC,
    const int outH,
    const int outW,
    const int channelMultiplier,
    const int filterH,
    const int filterW,
    __global const T *in0,
    __global const T* in1,
    __global T *out)
{
    for (int b = 0; b < batchSize; ++b)
    {
        for (int i = 0; i < outH; ++i)
        {
            for (int j = 0; j < outW; ++j)
            {
                for (int k = 0; k < inC; ++k)
                {
                    for (int q = 0; q < channelMultiplier; ++q)
                    {
                        for (int di = 0; di < filterH; ++di)
                        {
                            for (int dj = 0; dj < filterW; ++dj)
                            {
#if defined(USE_FORWARD)
                                out[b * outH * outW * inC * channelMultiplier + i * outW * inC * channelMultiplier + j * inC * channelMultiplier + k * channelMultiplier + q] += in1[di * filterW * inC * channelMultiplier + dj * inC * channelMultiplier + k * channelMultiplier + q] * in0[b * (outH + filterH - 1) * (outW + filterW - 1) * inC + (i + di) * (outW + filterW - 1) * inC +  (j + dj) * inC + k];
#elif defined(USE_BACKWARD)
#if defined(FOR_INPUT)
                                out[b * (outH + filterH - 1) * (outW + filterW - 1) * inC + (i + di) * (outW + filterW - 1) * inC +  (j + dj) * inC + k] += in1[di * filterW * inC * channelMultiplier + dj * inC * channelMultiplier + k * channelMultiplier + q] * in0[b * outH * outW * inC * channelMultiplier + i * outW * inC * channelMultiplier + j * inC * channelMultiplier + k * channelMultiplier + q];
#elif defined(FOR_FILTERS)
                                out[di * filterW * inC * channelMultiplier + dj * inC * channelMultiplier + k * channelMultiplier + q] += in1[b * (outH + filterH - 1) * (outW + filterW - 1) * inC + (i + di) * (outW + filterW - 1) * inC +  (j + dj) * inC + k] * in0[b * outH * outW * inC * channelMultiplier + i * outW * inC * channelMultiplier + j * inC * channelMultiplier + k * channelMultiplier + q];
#endif
#endif
                            }
                        }
                    }
                }
            }
        }
    }
}

)"