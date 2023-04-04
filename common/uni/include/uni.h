// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_UNI
#define _H_UNI

#include "sys.h"
#include "data_type.h"
#include "operator_type.h"
#include "parameter_spec.h"
#include "error.h"
#include "memory_cpu.h"
#include "affinity_policy.h"
#include "tensor_transpose.h"

#define UNUSED(x) (void)x
#define UNI_MIN(a, b) (((a) < (b)) ? (a) : (b))
#define UNI_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define UNI_ABS(a) (((a) > 0) ? (a) : (-1 * (a)))
#define UNI_SIGN(a) (((a) > 0) ? 1 : (((a) < 0) ? -1 : 0))
#define UNI_ALIGN(x, align) (((align) == 0) ? 0 : (((x) + (align)-1) / (align) * (align)))
#define UNI_F16_MIN -65504.0f
#define UNI_F16_MAX 65504.0f
#define UNI_RESERVE 9035
#define UNI_DYNAMIC_SHAPE 11
#endif
