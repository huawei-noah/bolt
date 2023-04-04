// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_BOLT_C_SIMPLIFY
#define _H_BOLT_C_SIMPLIFY

#ifdef __cplusplus
extern "C" {
#endif

int CreateInference(int useModelFileStream,
    const char *modelPath,
    const char *algorithmMapPath,
    AFFINITY_TYPE affinity,
    ModelHandle *inferenceHandle,
    ResultHandle *resultHandle);

int MallocTensorDesc(
    int num, char ***name, int **n, int **c, int **h, int **w, DATA_TYPE **dt, DATA_FORMAT **df);

int FreeTensorDesc(
    int num, char **name, int *n, int *c, int *h, int *w, DATA_TYPE *dt, DATA_FORMAT *df);

int CreateInputTensorDesc(ModelHandle inferenceHandle,
    int *num,
    char ***name,
    int **n,
    int **c,
    int **h,
    int **w,
    DATA_TYPE **dt,
    DATA_FORMAT **df);

int CreateOutputTensorDesc(ResultHandle resultHandle,
    int *num,
    char ***name,
    int **n,
    int **c,
    int **h,
    int **w,
    DATA_TYPE **dt,
    DATA_FORMAT **df);

int MallocTensor(int num,
    char **name,
    const int *n,
    const int *c,
    const int *h,
    const int *w,
    const DATA_TYPE *dt,
    const DATA_FORMAT *df,
    void ***data);

int FreeTensor(
    int num, char **name, int *n, int *c, int *h, int *w, DATA_TYPE *dt, DATA_FORMAT *df, void **data);

int TransformDataTypeAndFormat(ModelHandle handle,
    DATA_TYPE idt,
    DATA_FORMAT idf,
    int in,
    int ic,
    int ih,
    int iw,
    const void *input,
    DATA_TYPE odt,
    DATA_FORMAT odf,
    int on,
    int oc,
    int oh,
    int ow,
    void *output);

int InitTensor(int num,
    char **name,
    const int *n,
    const int *c,
    const int *h,
    const int *w,
    const DATA_TYPE *dt,
    const DATA_FORMAT *df,
    void **data,
    float value);

int LoadTensor(int num,
    char **name,
    const int *n,
    const int *c,
    const int *h,
    const int *w,
    const DATA_TYPE *dt,
    const DATA_FORMAT *df,
    void **data,
    const char *dataPath);

int PrintTensor(int num,
    char **name,
    const int *n,
    const int *c,
    const int *h,
    const int *w,
    const DATA_TYPE *dt,
    const DATA_FORMAT *df,
    void **data,
    const char *printPrefix,
    int printElementNum);
#ifdef __cplusplus
}
#endif
#endif
