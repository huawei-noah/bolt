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

int getValue(int4 v, int i) {
    if (i==0) return v.x;
    if (i==1) return v.y;
    if (i==2) return v.z;
    if (i==3) return v.w;
}

int4 setValue(int4 v, int i, int val) {
    if (i==0) v.x = val;
    if (i==1) v.y = val;
    if (i==2) v.z = val;
    if (i==3) v.w = val;
    return v;
}

int4 offsetToIndexes(int offset, const int4 strides) 
{     
    int4 indexes;                                            
    for (int q = 0; q < 4; ++q)                                      
    {       
        const int stride = getValue(strides, q);                                      
        if (stride != 0)                      
        {                                         
            indexes = setValue(indexes, q, offset / stride);     
            offset %= stride;                 
        }                                         
        else                                      
        {                                          
            indexes = setValue(indexes, q, 0);                       
        }                                         
    }                                             
    indexes.s3 += offset;
    return indexes;
}

int indexesToOffset(const int4 indexes, const int4 strides)
{
    int offset = 0;
    for (int q = 0; q < 4; ++q)                  
    {
        offset += getValue(indexes, q) * getValue(strides, q);
    }
    return offset;
}

__kernel void cumsumForward(const int x,
    const int y,
    const int z,
    const int dimension,
    const int size,
    const int str0,
    const int str1,
    const int str2,
    const int str3,
    __global const T *input,
    __global T *output)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    
    if (x == idx && y == idy && idz == z)
    {
        int4 indexes;
        int4 strides = (int4)(str0, str1, str2, str3);
        for (int q = 0; q < size; ++q)
        {
            indexes = offsetToIndexes(q, strides);
            const int index = getValue(indexes, dimension);
            if (index == 0)
            {
                output[q] = input[q];
            }
            else
            {
                indexes = setValue(indexes, dimension, index - 1);
                output[q] = input[q] + output[indexesToOffset(indexes, strides)];
            }
        }
    }
}

__kernel void cumsumBackward(const int x,
    const int y,
    const int z,
    const int dimension,
    const int size,
    const int str0,
    const int str1,
    const int str2,
    const int str3,
    __global const T *deltas,
    __global T *prevLayerDelta)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    
    if (x == idx && y == idy && idz == z)
    {
        int4 indexes;
        int4 strides = (int4)(str0, str1, str2, str3);
        int4 outputShape = (int4)(size / strides.s0, strides.s0 / strides.s1, strides.s1 / strides.s2, strides.s2);
        for (int q = size; q > 0; --q)
        {
            indexes = offsetToIndexes(q - 1, strides);
            const int index = getValue(indexes, dimension);
            if (index == getValue(outputShape, dimension) - 1)
            {
                prevLayerDelta[q - 1] += deltas[q - 1];
            }
            else
            {
                indexes = setValue(indexes, dimension, index + 1);
                prevLayerDelta[q - 1] += deltas[q - 1] + prevLayerDelta[indexesToOffset(indexes, strides)];
            }
        }
    }
}
)"
