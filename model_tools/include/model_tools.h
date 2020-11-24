// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MODEL_TOOLS
#define _H_MODEL_TOOLS

#include "tensor_desc.h"
#include "op_type.h"

#ifdef __cplusplus
extern "C" {
#endif

EE mt_create_model(ModelSpec *md);
EE mt_load(CI8 *dir, CI8 *mfn, ModelSpec *md);
#if defined(_USE_CAFFE) || defined(_USE_ONNX) || defined(_USE_TFLITE) || defined(_USE_TENSORFLOW)
EE mt_store(CI8 *dir, CI8 *mfn, const ModelSpec *md);
#endif
EE mt_destroy_model(ModelSpec *md);

#ifdef __cplusplus
}
#endif

#endif
