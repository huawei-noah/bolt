// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_CONVERTER
#define _H_CONVERTER
#include "type.h"
#include "error.h"
#include "model_tools.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _USE_CAFFE_MODEL
    EE mt_load_caffe(CI8* dir, CI8* mfn, ModelSpec* ms);
    EE mt_store_caffe(CI8* dir, CI8* mfn, const ModelSpec* ms);
    EE mt_to_caffe(CI8* dir, CI8* mtFn, CI8* caffeFn);
    EE mt_from_caffe(CI8* dir, CI8* caffeFn, CI8* mtFn);
    EE caffe_model_info_check(CI8* dir, CI8* mfn);
#endif

#ifdef _USE_TENSORFLOW_MODEL
    EE mt_load_tensorflow(CI8* dir, CI8* mfn, ModelSpec* ms);
    EE mt_store_tensorflow(CI8* dir, CI8* mfn, const ModelSpec* ms);
    EE mt_to_tf(CI8* dir, CI8* mtFn, CI8* tfFn);
    EE mt_from_tf(CI8* dir, CI8* tfFn, CI8* mtFn);
#endif

#ifdef _USE_ONNX_MODEL
    EE mt_load_onnx(CI8* dir, CI8* mfn, ModelSpec* ms, int removePreprocessOpNum, TensorDesc inputDesc);
    EE mt_store_onnx(CI8* dir, CI8* mfn, const ModelSpec* ms);
    EE mt_to_onnx(CI8* dir, CI8* mtFn, CI8* tfFn);
    EE mt_from_onnx(CI8* dir, CI8* tfFn, CI8* mtFn);
#endif

#ifdef _USE_TFLITE_MODEL
    EE mt_load_tflite(CI8* dir, CI8* mfn, ModelSpec* ms);
    EE mt_store_tflite(CI8* dir, CI8* mfn, const ModelSpec* ms);
    EE mt_to_tflite(CI8* dir, CI8* mtFn, CI8* tfFn);
    EE mt_from_tflite(CI8* dir, CI8* tfFn, CI8* mtFn);
#endif

#ifdef __cplusplus
}
#endif

#endif
