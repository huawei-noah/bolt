// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_IMAGE_CPU_FUNCTIONS
#define _H_IMAGE_CPU_FUNCTIONS

#include "error.h"
#include "uni.h"

template <RoundMode round_mode>
inline int round_d(const float& x)
{
    int ret = 0;
    switch (round_mode) {
        case ROUND_FLOOR:
            ret = floor(x);
            break;
        case ROUND_CEIL:
            ret = ceil(x);
            break;
        case ROUND_PREFER_FLOOR:
            ret = round(x);
            if (ret - x == 0.5) {
                ret -= 1;
            }
            break;
        case ROUND_PREFER_CEIL:
            ret = round(x);
            break;
        default:
            UNI_ERROR_LOG("Resize currently not support this round mode.\n");
            break;
    }
    return ret;
}

template <CoordinateTransMode trans_mode>
inline F32 coordinate_trans(const I32& x, const I32& iw, const I32& ow, float r0, const float& r1)
{
    F32 ret = 0;
    switch (trans_mode) {
        case COORDINATE_TRANS_HALF_PIXEL:
            ret = UNI_MAX(0, (x + 0.5f) * r0 - 0.5f);
            break;
        case COORDINATE_TRANS_PYTORCH_HALF_PIXEL:
            ret = (ow > 1) ? UNI_MAX(0.f, ((x + 0.5f) * r0 - 0.5)) : 0;
            break;
        case COORDINATE_TRANS_ALIGN_CORNERS:
            ret = x * r1;
            break;
        case COORDINATE_TRANS_ASYMMETRIC:
            ret = x * r0;
            break;
        default:
            UNI_ERROR_LOG("Resize currently not support this coordinate transformation "
                          "mode.\n");
            break;
    }
    return ret;
}
#endif
