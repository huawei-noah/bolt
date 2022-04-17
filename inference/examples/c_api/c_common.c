// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <stdlib.h>
#include <stdio.h>
#ifdef _USE_FP16
#include <arm_neon.h>
#endif
#include "../../examples/c_api/c_common.h"

void MallocTensorDesc(
    int num, char ***name, int **n, int **c, int **h, int **w, DATA_TYPE **dt, DATA_FORMAT **df)
{
    *name = (char **)malloc(sizeof(char *) * num);
    for (int i = 0; i < num; i++) {
        (*name)[i] = (char *)malloc(sizeof(char) * 128);
    }
    *n = (int *)malloc(sizeof(int) * num);
    *c = (int *)malloc(sizeof(int) * num);
    *h = (int *)malloc(sizeof(int) * num);
    *w = (int *)malloc(sizeof(int) * num);
    *dt = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * num);
    *df = (DATA_FORMAT *)malloc(sizeof(DATA_FORMAT) * num);
}

void FreeTensorDesc(
    int num, char **name, int *n, int *c, int *h, int *w, DATA_TYPE *dt, DATA_FORMAT *df)
{
    for (int i = 0; i < num; i++) {
        free(name[i]);
    }
    free(name);
    free(n);
    free(c);
    free(h);
    free(w);
    free(dt);
    free(df);
}

void CreateInputTensorDesc(ModelHandle inferenceHandle,
    int *num,
    char ***name,
    int **n,
    int **c,
    int **h,
    int **w,
    DATA_TYPE **dt,
    DATA_FORMAT **df)
{
    *num = GetNumInputsFromModel(inferenceHandle);
    MallocTensorDesc(*num, name, n, c, h, w, dt, df);
    GetInputDataInfoFromModel(inferenceHandle, *num, *name, *n, *c, *h, *w, *dt, *df);
}

void CreateOutputTensorDesc(ResultHandle resultHandle,
    int *num,
    char ***name,
    int **n,
    int **c,
    int **h,
    int **w,
    DATA_TYPE **dt,
    DATA_FORMAT **df)
{
    *num = GetNumOutputsFromResultHandle(resultHandle);
    MallocTensorDesc(*num, name, n, c, h, w, dt, df);
    GetOutputDataInfoFromResultHandle(resultHandle, *num, *name, *n, *c, *h, *w, *dt, *df);
}

void MallocTensor(int num,
    char **name,
    const int *n,
    const int *c,
    const int *h,
    const int *w,
    const DATA_TYPE *dt,
    const DATA_FORMAT *df,
    void ***data)
{
    *data = (void **)malloc(sizeof(void *) * num);
    for (int i = 0; i < num; i++) {
        int length = n[i] * c[i] * h[i] * w[i];
        switch (dt[i]) {
            case FP_32: {
                (*data)[i] = malloc(sizeof(float) * length);
                break;
            }
            case UINT_32: {
                (*data)[i] = malloc(sizeof(unsigned) * length);
                break;
            }
#ifdef _USE_FP16
            case FP_16: {
                (*data)[i] = malloc(sizeof(__fp16) * length);
                break;
            }
#endif
            default:
                printf("[ERROR] unsupported data precision in %s\n", __func__);
                exit(1);
        }
    }
}

void FreeTensor(
    int num, char **name, int *n, int *c, int *h, int *w, DATA_TYPE *dt, DATA_FORMAT *df, void **data)
{
    FreeTensorDesc(num, name, n, c, h, w, dt, df);
    for (int i = 0; i < num; i++) {
        free(data[i]);
    }
    free(data);
}

void CreateInference(int useModelFileStream,
    const char *modelPath,
    const char *algorithmMapPath,
    AFFINITY_TYPE affinity,
    ModelHandle *inferenceHandle,
    ResultHandle *resultHandle)
{
    if (useModelFileStream) {
        *inferenceHandle = CreateModelWithFileStream(modelPath, affinity, algorithmMapPath);
    } else {
        *inferenceHandle = CreateModel(modelPath, affinity, algorithmMapPath);
    }

    char **in_name;
    int *in_n, *in_c, *in_h, *in_w;
    DATA_TYPE *in_dt;
    DATA_FORMAT *in_df;
    int in_num = GetNumInputsFromModel(*inferenceHandle);
    MallocTensorDesc(in_num, &in_name, &in_n, &in_c, &in_h, &in_w, &in_dt, &in_df);

    GetInputDataInfoFromModel(
        *inferenceHandle, in_num, in_name, in_n, in_c, in_h, in_w, in_dt, in_df);

    PrepareModel(
        *inferenceHandle, in_num, (const char **)in_name, in_n, in_c, in_h, in_w, in_dt, in_df);

    *resultHandle = AllocAllResultHandle(*inferenceHandle);

    for (int i = 0; i < in_num; i++) {
        free(in_name[i]);
    }
    free(in_name);
    free(in_n);
    free(in_c);
    free(in_h);
    free(in_w);
    free(in_dt);
    free(in_df);
}
