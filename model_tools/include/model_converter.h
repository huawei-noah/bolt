// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MODEL_CONVERTER
#define _H_MODEL_CONVERTER

#include <string>
#include "model_spec.h"

#ifdef _USE_CAFFE
EE caffe_converter(std::string dir, std::string mfn, ModelSpec *ms);
#endif

#ifdef _USE_ONNX
EE onnx_converter(std::string dir, std::string mfn, int removePreprocessOpNum, ModelSpec *ms);
#endif

#ifdef _USE_TFLITE
EE tflite_converter(std::string dir, std::string mfn, ModelSpec *ms);
#endif

#ifdef _USE_TENSORFLOW
EE tensorflow_converter(std::string dir, std::string mfn, ModelSpec *ms);
#endif
#endif
