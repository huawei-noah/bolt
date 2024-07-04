// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "bolt_c_common.h"
#include "../api/c/bolt_simplify.h"

#include "file.h"
#include "data_type.h"
#include "memory_cpu.h"

int MallocTensorDesc(
    int num, char ***name, int **n, int **c, int **h, int **w, DATA_TYPE **dt, DATA_FORMAT **df)
{
    UNI_DEBUG_LOG("C API %s(%d, %p, %p, %p, %p, %p, %p, %p)...\n", __FUNCTION__, num, name, n, c, h,
        w, dt, df);
    int ret = 0;
    if (num > 0) {
        *name = (char **)UNI_MALLOC(sizeof(char *) * num);
        for (int i = 0; i < num; i++) {
            (*name)[i] = (char *)UNI_MALLOC(sizeof(char) * 128);
        }
        *n = (int *)UNI_MALLOC(sizeof(int) * num);
        *c = (int *)UNI_MALLOC(sizeof(int) * num);
        *h = (int *)UNI_MALLOC(sizeof(int) * num);
        *w = (int *)UNI_MALLOC(sizeof(int) * num);
        *dt = (DATA_TYPE *)UNI_MALLOC(sizeof(DATA_TYPE) * num);
        *df = (DATA_FORMAT *)UNI_MALLOC(sizeof(DATA_FORMAT) * num);
    }
    UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
    return ret;
}

int FreeTensorDesc(
    int num, char **name, int *n, int *c, int *h, int *w, DATA_TYPE *dt, DATA_FORMAT *df)
{
    UNI_DEBUG_LOG("C API %s(%d, %p, %p, %p, %p, %p, %p, %p)...\n", __FUNCTION__, num, name, n, c, h,
        w, dt, df);
    int ret = 0;
    if (num > 0) {
        for (int i = 0; i < num; i++) {
            UNI_FREE(name[i]);
        }
        UNI_FREE(name);
        UNI_FREE(n);
        UNI_FREE(c);
        UNI_FREE(h);
        UNI_FREE(w);
        UNI_FREE(dt);
        UNI_FREE(df);
    }
    UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
    return ret;
}

int CreateInputTensorDesc(ModelHandle inferenceHandle,
    int *num,
    char ***name,
    int **n,
    int **c,
    int **h,
    int **w,
    DATA_TYPE **dt,
    DATA_FORMAT **df)
{
    UNI_DEBUG_LOG("C API %s(%p, %p, %p, %p, %p, %p, %p, %p, %p)...\n", __FUNCTION__,
        inferenceHandle, num, name, n, c, h, w, dt, df);
    int ret = 1;
    if (inferenceHandle != NULL) {
        *num = GetNumInputsFromModel(inferenceHandle);
        ret = MallocTensorDesc(*num, name, n, c, h, w, dt, df);
        if (ret == 0) {
            GetInputDataInfoFromModel(inferenceHandle, *num, *name, *n, *c, *h, *w, *dt, *df);
        }
    }
    UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
    return ret;
}

int CreateOutputTensorDesc(ResultHandle resultHandle,
    int *num,
    char ***name,
    int **n,
    int **c,
    int **h,
    int **w,
    DATA_TYPE **dt,
    DATA_FORMAT **df)
{
    UNI_DEBUG_LOG("C API %s(%p, %p, %p, %p, %p, %p, %p, %p, %p)...\n", __FUNCTION__, resultHandle,
        num, name, n, c, h, w, dt, df);
    int ret = 1;
    if (resultHandle != NULL) {
        *num = GetNumOutputsFromResultHandle(resultHandle);
        ret = MallocTensorDesc(*num, name, n, c, h, w, dt, df);
        if (ret == 0) {
            GetOutputDataInfoFromResultHandle(resultHandle, *num, *name, *n, *c, *h, *w, *dt, *df);
        }
    }
    UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
    return ret;
}

int MallocTensor(int num,
    char **name,
    const int *n,
    const int *c,
    const int *h,
    const int *w,
    const DATA_TYPE *dt,
    const DATA_FORMAT *df,
    void ***data)
{
    UNI_DEBUG_LOG("C API %s(%d, %p, %p, %p, %p, %p, %p, %p, %p)...\n", __FUNCTION__, num, name, n,
        c, h, w, dt, df, data);
    int ret = 0;
    if (num > 0) {
        *data = (void **)UNI_MALLOC(sizeof(void *) * num);
        for (int i = 0; i < num; i++) {
            int length = n[i] * c[i] * h[i] * w[i];
            (*data)[i] = UNI_MALLOC(GetDataTypeSize(dt[i]) * length);
        }
    }
    UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
    return ret;
}

int FreeTensor(
    int num, char **name, int *n, int *c, int *h, int *w, DATA_TYPE *dt, DATA_FORMAT *df, void **data)
{
    UNI_DEBUG_LOG("C API %s(%d, %p, %p, %p, %p, %p, %p, %p, %p)...\n", __FUNCTION__, num, name, n,
        c, h, w, dt, df, data);
    int ret = 0;
    if (num > 0) {
        for (int i = 0; i < num; i++) {
            UNI_FREE(data[i]);
        }
        UNI_FREE(data);
    }
    UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
    return ret;
}

int CreateInference(int useModelFileStream,
    const char *modelPath,
    const char *algorithmMapPath,
    AFFINITY_TYPE affinity,
    ModelHandle *inferenceHandle,
    ResultHandle *resultHandle)
{
    UNI_DEBUG_LOG("C API %s(%d, %p, %p, %d, %p, %p)...\n", __FUNCTION__, useModelFileStream,
        modelPath, algorithmMapPath, affinity, inferenceHandle, resultHandle);
    int ret = 0;
    if (inferenceHandle == NULL || resultHandle == NULL) {
        ret = 1;
        UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
        return ret;
    }
    if (useModelFileStream) {
        *inferenceHandle = CreateModelWithFileStream(modelPath, affinity, algorithmMapPath);
    } else {
        *inferenceHandle = CreateModel(modelPath, affinity, algorithmMapPath);
    }
    if (*inferenceHandle == NULL) {
        ret = 1;
        UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
        return ret;
    }
    char **in_name;
    int *in_n, *in_c, *in_h, *in_w;
    DATA_TYPE *in_dt;
    DATA_FORMAT *in_df;
    int in_num = GetNumInputsFromModel(*inferenceHandle);
    ret = MallocTensorDesc(in_num, &in_name, &in_n, &in_c, &in_h, &in_w, &in_dt, &in_df);
    if (ret != 0) {
        UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
        return ret;
    }
    GetInputDataInfoFromModel(
        *inferenceHandle, in_num, in_name, in_n, in_c, in_h, in_w, in_dt, in_df);
    PrepareModel(
        *inferenceHandle, in_num, (const char **)in_name, in_n, in_c, in_h, in_w, in_dt, in_df);
    *resultHandle = AllocAllResultHandle(*inferenceHandle);
    if (*resultHandle == NULL) {
        ret = 1;
        UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
        return ret;
    }
    ret = FreeTensorDesc(in_num, in_name, in_n, in_c, in_h, in_w, in_dt, in_df);
    UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
    return ret;
}

#ifndef _USE_LITE
template <typename T1, typename T2>
void TransformNCHWCxToNCHW(DATA_FORMAT idf,
    int in,
    int ic,
    int ih,
    int iw,
    const T1 *src,
    DATA_FORMAT odf,
    int on,
    int oc,
    int oh,
    int ow,
    T2 *dst)
{
    int align = 1;
    if (idf == NCHWC8) {
        align = 8;
    }
    if (idf == NCHWC4) {
        align = 4;
    }
    ic /= align;
    for (int n = 0, i = 0; n < on; n++) {
        for (int c = 0; c < oc; c++) {
            for (int h = 0; h < oh; h++) {
                for (int w = 0; w < ow; w++, i++) {
                    int c1 = c / align;
                    int c2 = c % align;
                    int j = (((n * ic + c1) * ih + h) * iw + w) * align + c2;
                    if (sizeof(T1) == sizeof(unsigned short)) {
                        dst[i] = float16ToFloat32(src[j]);
                    } else {
                        dst[i] = src[j];
                    }
                }
            }
        }
    }
}
#endif

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
    void *output)
{
    UNI_DEBUG_LOG("C API %s(%p, %s, %s, %d, %d, %d, %d, %p, %s, %s, %d, %d, %d, %d, %p)...\n",
        __FUNCTION__, handle, GetDataTypeString()[idt], GetDataFormatString()[idf], in, ic, ih, iw,
        input, GetDataTypeString()[odt], GetDataFormatString()[odf], on, oc, oh, ow, output);
#ifndef _USE_LITE
    int ret = 0;
    if (odf != NCHW) {
        ret = 1;
        UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
        return ret;
    }
    if (odt != FP_32) {
        ret = 1;
        UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
        return ret;
    }
    size_t length = on * oc * oh * ow;
    if (idf != NCHWC8 && idf != NCHWC4) {
        ret = TransformDataType(handle, idt, input, NULL, odt, output, NULL, length);
        UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
        return ret;
    }
    if (idt == FP_32) {
        TransformNCHWCxToNCHW<float, float>(
            idf, in, ic, ih, iw, (const float *)input, odf, on, oc, oh, ow, (float *)output);
        UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
        return ret;
    }
    if (idt == FP_16) {
        TransformNCHWCxToNCHW<unsigned short, float>(idf, in, ic, ih, iw,
            (const unsigned short *)input, odf, on, oc, oh, ow, (float *)output);
        UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
        return ret;
    }
#else
    int ret = 1;
#endif
    UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
    return ret;
}

int InitTensor(int num,
    char **name,
    const int *n,
    const int *c,
    const int *h,
    const int *w,
    const DATA_TYPE *dt,
    const DATA_FORMAT *df,
    void **data,
    float value)
{
    UNI_DEBUG_LOG("C API %s(%d, %p, %p, %p, %p, %p, %p, %p, %p, %f)...\n", __FUNCTION__, num, name,
        n, c, h, w, dt, df, data, value);
    int ret = 0;
    for (int i = 0; i < num; i++) {
        U32 length = n[i] * c[i] * h[i] * w[i];
        DataType type = DATA_TYPE2DataType(dt[i]);
        UNI_INIT(length, type, value, data[i]);
    }
    UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
    return ret;
}

int LoadTensor(int num,
    char **name,
    const int *n,
    const int *c,
    const int *h,
    const int *w,
    const DATA_TYPE *dt,
    const DATA_FORMAT *df,
    void **data,
    const char *directory)
{
    UNI_DEBUG_LOG("C API %s(%d, %p, %p, %p, %p, %p, %p, %p, %p, %s)...\n", __FUNCTION__, num, name,
        n, c, h, w, dt, df, data, directory);
#ifndef _USE_LITE
    int ret = 0;
    std::string path;
    for (int i = 0; i < num; i++) {
        U32 length = n[i] * c[i] * h[i] * w[i];
        DataType type = DATA_TYPE2DataType(dt[i]);
        if (is_directory(directory)) {
            path = std::string(directory) + "/" + std::string(name[i]) + ".txt";
        } else {
            path = directory;
        }
        if (type == DT_F32) {
            CHECK_STATUS(load_floats(path.c_str(), (float *)data[i], length));
        } else if (type == DT_I32 || type == DT_U32) {
            std::vector<float> tmp(length);
            CHECK_STATUS(load_floats(path.c_str(), tmp.data(), length));
            transformFromFloat(type, tmp.data(), data[i], length);
        } else {
            UNI_INIT(length, type, 1, data[i]);
        }
    }
#else
    int ret = 1;
#endif
    UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
    return ret;
}

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
    int printElementNum)
{
    UNI_DEBUG_LOG("C API %s(%d, %p, %p, %p, %p, %p, %p, %p, %p, %s, %d)...\n", __FUNCTION__, num,
        name, n, c, h, w, dt, df, data, printPrefix, printElementNum);
    int ret = 0;
    for (int i = 0; i < num; i++) {
        printf("%sname:%s type:%s format:%s(%d,%d,%d,%d) data:", printPrefix, name[i],
            GetDataTypeString()[dt[i]], GetDataFormatString()[df[i]], n[i], c[i], h[i], w[i]);
        if (data[i] == NULL) {
            printf("\n");
            continue;
        }
        int length = n[i] * c[i] * h[i] * w[i];
        DataType type = DATA_TYPE2DataType(dt[i]);
        std::vector<float> tmp(length);
        transformToFloat(type, data[i], tmp.data(), length);
        float sum = 0;
        for (int j = 0; j < length; j++) {
            sum += tmp[j];
            if (j < printElementNum) {
                printf("%f ", tmp[j]);
            }
        }
        printf("sum:%f\n", sum);
        fflush(stdout);
    }
    UNI_DEBUG_LOG("C API %s(%d) end.\n", __FUNCTION__, ret);
    return ret;
}
