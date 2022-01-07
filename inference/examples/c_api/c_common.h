// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_C_COMMON
#define _H_C_COMMON

#include <stdlib.h>
#include "../api/c/bolt.h"

void MallocTensorDesc(
    int num, char ***name, int **n, int **c, int **h, int **w, DATA_TYPE **dt, DATA_FORMAT **df);

void FreeTensorDesc(
    int num, char **name, int *n, int *c, int *h, int *w, DATA_TYPE *dt, DATA_FORMAT *df);

void CreateInputTensorDesc(ModelHandle inferenceHandle,
    int *num,
    char ***name,
    int **n,
    int **c,
    int **h,
    int **w,
    DATA_TYPE **dt,
    DATA_FORMAT **df);

void CreateOutputTensorDesc(ResultHandle resultHandle,
    int *num,
    char ***name,
    int **n,
    int **c,
    int **h,
    int **w,
    DATA_TYPE **dt,
    DATA_FORMAT **df);

void MallocTensor(int num,
    char **name,
    const int *n,
    const int *c,
    const int *h,
    const int *w,
    const DATA_TYPE *dt,
    const DATA_FORMAT *df,
    void ***data);

void FreeTensor(
    int num, char **name, int *n, int *c, int *h, int *w, DATA_TYPE *dt, DATA_FORMAT *df, void **data);

void CreateInference(int useModelFileStream,
    const char *modelPath,
    const char *algorithmMapPath,
    AFFINITY_TYPE affinity,
    ModelHandle *inferenceHandle,
    ResultHandle *resultHandle,
    int *inputNum,
    char ***inputName);
#endif
