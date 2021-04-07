// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifdef _USE_JNI
#include "BoltModel.h"
#include "cnn.h"
#include "../api/c/bolt.h"

struct ModelHandleInfo {
    void *ms;
    void *cnn;
    DEVICE_TYPE deviceType;
    void *algoPath;
    bool useFileStream;
};

typedef struct DataDesc {
    U32 dims[6] = {0};
    char name[NAME_LEN] = {0};
    DataType dt;
    DataFormat df;
    void *dataPtr;
} DataDesc;

typedef struct {
    U32 num_outputs;
    DataDesc *outputArr;
    DEVICE_TYPE deviceType;
} ResultHandleInner;

AFFINITY_TYPE str2AFFINITY_TYPE(std::string affinity_str)
{
    AFFINITY_TYPE ret = CPU_HIGH_PERFORMANCE;
    if (affinity_str == "CPU_HIGH_PERFORMANCE") {
        ret = CPU_HIGH_PERFORMANCE;
    } else if (affinity_str == "CPU_LOW_POWER") {
        ret = CPU_LOW_POWER;
    } else if (affinity_str == "GPU") {
        ret = GPU;
    } else {
        UNI_ERROR_LOG("unsupported JNI CPU affinity setting %s\n", affinity_str.c_str());
    }
    return ret;
}

DEVICE_TYPE str2DEVICE_TYPE(std::string device_str)
{
    DEVICE_TYPE ret = CPU_ARM_V8;
    if (device_str == "CPU_ARM_V7") {
        ret = CPU_ARM_V7;
    } else if (device_str == "CPU_ARM_V8") {
        ret = CPU_ARM_V8;
    } else if (device_str == "CPU_ARM_A55") {
        ret = CPU_ARM_A55;
    } else if (device_str == "CPU_ARM_A76") {
        ret = CPU_ARM_A76;
    } else if (device_str == "GPU_MALI") {
        ret = GPU_MALI;
    } else if (device_str == "CPU_X86_AVX2") {
        ret = CPU_X86_AVX2;
    } else if (device_str == "CPU_SERIAL") {
        ret = CPU_SERIAL;
    } else {
        UNI_ERROR_LOG("unsupported JNI device setting %s\n", device_str.c_str());
    }
    return ret;
}

DATA_TYPE str2DATA_TYPE(std::string data_type)
{
    DATA_TYPE ret = FP_32;
    if (data_type == "FP32") {
        ret = FP_32;
#ifdef __aarch64__
    } else if (data_type == "FP16") {
        ret = FP_16;
#endif
    } else if (data_type == "INT32") {
        ret = INT_32;
    } else if (data_type == "UINT32") {
        ret = UINT_32;
    } else {
        UNI_ERROR_LOG("unsupported JNI data type setting %s\n", data_type.c_str());
    }
    return ret;
}

DATA_FORMAT str2DATA_FORMAT(std::string data_format)
{
    DATA_FORMAT ret = NCHW;
    if (data_format == "NCHW") {
        ret = NCHW;
    } else if (data_format == "NHWC") {
        ret = NHWC;
    } else if (data_format == "MTK") {
        ret = MTK;
    } else if (data_format == "NORMAL") {
        ret = NORMAL;
    } else {
        UNI_ERROR_LOG("unsupported JNI data format setting %s\n", data_format.c_str());
    }
    return ret;
}

std::string DataFormat2str(DataFormat data_format)
{
    std::string ret = "NCHW";
    switch (data_format) {
        case DF_NCHW:
            ret = "NCHW";
            break;
        case DF_NCHWC8:
            ret = "NCHWC8";
            break;
        case DF_NHWC:
            ret = "NHWC";
            break;
        case DF_MTK:
            ret = "MTK";
            break;
        case DF_NORMAL:
            ret = "NORMAL";
            break;
        default:
            UNI_ERROR_LOG("unsupported JNI data format setting %d\n", data_format);
    }
    return ret;
}

void getInputParameters(JNIEnv *env,
    jint num,
    jobjectArray input_names,
    char ***data_name_ptr,
    jintArray n,
    int **data_n_ptr,
    jintArray c,
    int **data_c_ptr,
    jintArray h,
    int **data_h_ptr,
    jintArray w,
    int **data_w_ptr,
    jobjectArray dt_input,
    DATA_TYPE **data_dt_ptr,
    jobjectArray df_input,
    DATA_FORMAT **data_df_ptr)
{
    if (env->GetArrayLength(input_names) != num) {
        UNI_ERROR_LOG("input name array length %d is not equal to input num %d\n",
            env->GetArrayLength(input_names), num);
    }
    if (env->GetArrayLength(n) != num) {
        UNI_ERROR_LOG(
            "input N array length %d is not equal to input num %d\n", env->GetArrayLength(n), num);
    }
    if (env->GetArrayLength(c) != num) {
        UNI_ERROR_LOG(
            "input C array length %d is not equal to input num %d\n", env->GetArrayLength(c), num);
    }
    if (env->GetArrayLength(h) != num) {
        UNI_ERROR_LOG(
            "input H array length %d is not equal to input num %d\n", env->GetArrayLength(h), num);
    }
    if (env->GetArrayLength(w) != num) {
        UNI_ERROR_LOG(
            "input W array length %d is not equal to input num %d\n", env->GetArrayLength(w), num);
    }
    if (env->GetArrayLength(dt_input) != num) {
        UNI_ERROR_LOG("input DataType array length %d is not equal to input num %d\n",
            env->GetArrayLength(dt_input), num);
    }
    if (env->GetArrayLength(df_input) != num) {
        UNI_ERROR_LOG("input DataFormat array length %d is not equal to input num %d\n",
            env->GetArrayLength(df_input), num);
    }
    int *data_n = (int *)malloc(num * sizeof(int));
    int *data_c = (int *)malloc(num * sizeof(int));
    int *data_h = (int *)malloc(num * sizeof(int));
    int *data_w = (int *)malloc(num * sizeof(int));
    char **data_name = (char **)malloc(num * sizeof(char *));
    DATA_TYPE *data_dt = (DATA_TYPE *)malloc(num * sizeof(DATA_TYPE));
    DATA_FORMAT *data_df = (DATA_FORMAT *)malloc(num * sizeof(DATA_FORMAT));
    jint *curArray_n = env->GetIntArrayElements(n, 0);
    jint *curArray_c = env->GetIntArrayElements(c, 0);
    jint *curArray_h = env->GetIntArrayElements(h, 0);
    jint *curArray_w = env->GetIntArrayElements(w, 0);
    for (int i = 0; i < num; i++) {
        data_n[i] = curArray_n[i];
        data_c[i] = curArray_c[i];
        data_h[i] = curArray_h[i];
        data_w[i] = curArray_w[i];

        jstring cur_str = (jstring)(env->GetObjectArrayElement(input_names, i));
        const char *cur_str_ptr = env->GetStringUTFChars(cur_str, 0);
        int length = strlen(cur_str_ptr);
        data_name[i] = (char *)malloc(sizeof(char) * (length + 1));
        UNI_MEMCPY(data_name[i], cur_str_ptr, length);
        data_name[i][length] = '\0';

        jstring tmp_str_dt = (jstring)(env->GetObjectArrayElement(dt_input, i));
        const char *tmp_str_dt_ptr = env->GetStringUTFChars(tmp_str_dt, 0);
        std::string cur_tmp_str_dt = tmp_str_dt_ptr;
        data_dt[i] = str2DATA_TYPE(cur_tmp_str_dt);

        jstring tmp_str_df = (jstring)(env->GetObjectArrayElement(df_input, i));
        const char *tmp_str_df_ptr = env->GetStringUTFChars(tmp_str_df, 0);
        std::string cur_tmp_str_df = tmp_str_df_ptr;
        data_df[i] = str2DATA_FORMAT(cur_tmp_str_df);

        env->ReleaseStringUTFChars(cur_str, cur_str_ptr);
        env->DeleteLocalRef(cur_str);
        env->ReleaseStringUTFChars(tmp_str_dt, tmp_str_dt_ptr);
        env->ReleaseStringUTFChars(tmp_str_df, tmp_str_df_ptr);
        env->DeleteLocalRef(tmp_str_dt);
        env->DeleteLocalRef(tmp_str_df);
    }
    env->ReleaseIntArrayElements(n, curArray_n, 0);
    env->ReleaseIntArrayElements(c, curArray_c, 0);
    env->ReleaseIntArrayElements(h, curArray_h, 0);
    env->ReleaseIntArrayElements(w, curArray_w, 0);
    *data_name_ptr = data_name;
    *data_n_ptr = data_n;
    *data_c_ptr = data_c;
    *data_h_ptr = data_h;
    *data_w_ptr = data_w;
    *data_dt_ptr = data_dt;
    *data_df_ptr = data_df;
}

int calculateLength(int *array, int num)
{
    int length = 0;
    for (int j = 0; j < num; j++) {
        if (array[j] == 0) {
            break;
        } else {
            if (length == 0) {
                length = array[j];
            } else {
                length *= array[j];
            }
        }
    }
    return length;
}

extern "C" JNIEXPORT jlong JNICALL BOLT_JNI_PREFIX(BoltModel_createModel)(
    JNIEnv *env, jobject, jstring modelPath, jstring affinity)
{
    const char *modelPathPtr = env->GetStringUTFChars(modelPath, JNI_FALSE);
    const char *affinityPtr = env->GetStringUTFChars(affinity, JNI_FALSE);
    std::string affinity_str = (std::string)affinityPtr;
    AFFINITY_TYPE affinity_cur = str2AFFINITY_TYPE(affinity_str);
    long modelAddr = (long)CreateModel(modelPathPtr, affinity_cur, NULL);
    ModelHandleInfo *ihInfo = (ModelHandleInfo *)modelAddr;
    if (nullptr == ihInfo->cnn) {
        UNI_ERROR_LOG("Bolt instance not created\n");
        modelAddr = 0;
    }
    env->ReleaseStringUTFChars(modelPath, modelPathPtr);
    env->ReleaseStringUTFChars(affinity, affinityPtr);
    return modelAddr;
}

extern "C" JNIEXPORT jlong JNICALL BOLT_JNI_PREFIX(BoltModel_cloneModel)(
    JNIEnv *env, jobject, jlong modelAddr)
{
    ModelHandle handle = (ModelHandle)modelAddr;
    ModelHandle cloneHandle = CloneModel(handle);
    long ret = (long)cloneHandle;
    return ret;
}

extern "C" JNIEXPORT void JNICALL BOLT_JNI_PREFIX(BoltModel_prepareModel)(JNIEnv *env,
    jobject,
    jlong modelAddr,
    jint num_input,
    jobjectArray input_names,
    jintArray n,
    jintArray c,
    jintArray h,
    jintArray w,
    jobjectArray dt_input,
    jobjectArray df_input)
{
    ModelHandle ih = (ModelHandle)modelAddr;
    char **data_name = nullptr;
    int *data_n = nullptr;
    int *data_c = nullptr;
    int *data_h = nullptr;
    int *data_w = nullptr;
    DATA_TYPE *data_dt = nullptr;
    DATA_FORMAT *data_df = nullptr;
    getInputParameters(env, num_input, input_names, &data_name, n, &data_n, c, &data_c, h, &data_h,
        w, &data_w, dt_input, &data_dt, df_input, &data_df);

    PrepareModel(
        ih, num_input, (const char **)data_name, data_n, data_c, data_h, data_w, data_dt, data_df);

    free(data_n);
    free(data_c);
    free(data_h);
    free(data_w);
    for (int i = 0; i < num_input; i++) {
        free(data_name[i]);
    }
    free(data_name);
    free(data_dt);
    free(data_df);
}

extern "C" JNIEXPORT void JNICALL BOLT_JNI_PREFIX(BoltModel_resizeModelInput)(JNIEnv *env,
    jobject,
    jlong modelAddr,
    jint num_input,
    jobjectArray input_names,
    jintArray n,
    jintArray c,
    jintArray h,
    jintArray w,
    jobjectArray dt_input,
    jobjectArray df_input)
{
    ModelHandle ih = (ModelHandle)modelAddr;
    char **data_name = nullptr;
    int *data_n = nullptr;
    int *data_c = nullptr;
    int *data_h = nullptr;
    int *data_w = nullptr;
    DATA_TYPE *data_dt = nullptr;
    DATA_FORMAT *data_df = nullptr;
    getInputParameters(env, num_input, input_names, &data_name, n, &data_n, c, &data_c, h, &data_h,
        w, &data_w, dt_input, &data_dt, df_input, &data_df);

    ResizeModelInput(
        ih, num_input, (const char **)data_name, data_n, data_c, data_h, data_w, data_dt, data_df);

    free(data_n);
    free(data_c);
    free(data_h);
    free(data_w);
    for (int i = 0; i < num_input; i++) {
        free(data_name[i]);
    }
    free(data_name);
    free(data_dt);
    free(data_df);
}

extern "C" JNIEXPORT jlong JNICALL BOLT_JNI_PREFIX(BoltModel_allocAllResultHandle)(
    JNIEnv *, jobject, jlong modelAddr)
{
    ModelHandle ih = (ModelHandle)modelAddr;
    ResultHandle ir = AllocAllResultHandle(ih);
    return (long)ir;
}

extern "C" JNIEXPORT jlong JNICALL BOLT_JNI_PREFIX(BoltModel_allocSpecificResultHandle)(
    JNIEnv *env, jobject, jlong modelAddr, jint num_outputs, jobjectArray outputNames)
{
    if (env->GetArrayLength(outputNames) != num_outputs) {
        UNI_ERROR_LOG("output name array length %d is not equal to output num %d\n",
            env->GetArrayLength(outputNames), num_outputs);
    }
    ModelHandle ih = (ModelHandle)modelAddr;
    char **output_names_ptr = (char **)malloc(sizeof(char *) * num_outputs);
    for (int i = 0; i < num_outputs; i++) {
        jstring cur_str = (jstring)(env->GetObjectArrayElement(outputNames, i));
        const char *cur_str_ptr = env->GetStringUTFChars(cur_str, 0);
        int length = strlen(cur_str_ptr);
        output_names_ptr[i] = (char *)malloc(sizeof(char) * (length + 1));
        UNI_MEMCPY(output_names_ptr[i], cur_str_ptr, length);
        output_names_ptr[i][length] = '\0';

        env->ReleaseStringUTFChars(cur_str, cur_str_ptr);
        env->DeleteLocalRef(cur_str);
    }
    ResultHandle ir = AllocSpecificResultHandle(ih, num_outputs, (const char **)output_names_ptr);

    for (int i = 0; i < num_outputs; i++) {
        free(output_names_ptr[i]);
    }
    free(output_names_ptr);
    return (long)ir;
}

extern "C" JNIEXPORT void JNICALL BOLT_JNI_PREFIX(BoltModel_setRuntimeDeviceJNI)(
    JNIEnv *env, jobject, jlong modelAddr, jint cpu_id, jstring device)
{
    ModelHandle ih = (ModelHandle)modelAddr;
    const char *devicePtr = env->GetStringUTFChars(device, JNI_FALSE);
    std::string device_str = (std::string)devicePtr;
    DEVICE_TYPE device_cur = str2DEVICE_TYPE(device_str);
    SetRuntimeDevice(ih, cpu_id, device_cur);
    env->ReleaseStringUTFChars(device, devicePtr);
}

extern "C" JNIEXPORT void JNICALL BOLT_JNI_PREFIX(BoltModel_setRuntimeDeviceDynamicJNI)(
    JNIEnv *env, jobject, jlong modelAddr)
{
    ModelHandle ih = (ModelHandle)modelAddr;
    SetRuntimeDeviceDynamic(ih);
}

extern "C" JNIEXPORT void JNICALL BOLT_JNI_PREFIX(BoltModel_runModel)(JNIEnv *env,
    jobject,
    jlong modelAddr,
    jlong ResultHandleAddr,
    jint num_input,
    jobjectArray input_names,
    jobjectArray inputData)
{
    if (env->GetArrayLength(input_names) != num_input) {
        UNI_ERROR_LOG("input name array length %d is not equal to input num %d\n",
            env->GetArrayLength(input_names), num_input);
    }
    if (env->GetArrayLength(inputData) != num_input) {
        UNI_ERROR_LOG("input data array length %d is not equal to input num %d\n",
            env->GetArrayLength(inputData), num_input);
    }
    ModelHandle ih = (ModelHandle)modelAddr;
    ResultHandle ir = (ResultHandle)ResultHandleAddr;

    ModelHandleInfo *ihInfo = (ModelHandleInfo *)ih;
    CNN *cnn = (CNN *)ihInfo->cnn;
    std::map<std::string, std::shared_ptr<Tensor>> inMap = cnn->get_input();

    char **input_names_ptr = (char **)malloc(sizeof(char *) * num_input);
    void **mem_ptr = (void **)malloc(sizeof(void *) * num_input);
    for (int i = 0; i < num_input; i++) {
        jstring cur_str = (jstring)(env->GetObjectArrayElement(input_names, i));
        const char *cur_str_ptr = env->GetStringUTFChars(cur_str, 0);
        int length = strlen(cur_str_ptr);
        input_names_ptr[i] = (char *)malloc(sizeof(char) * (length + 1));
        UNI_MEMCPY(input_names_ptr[i], cur_str_ptr, length);
        input_names_ptr[i][length] = '\0';
        env->ReleaseStringUTFChars(cur_str, cur_str_ptr);
        env->DeleteLocalRef(cur_str);

        jfloatArray curArray = static_cast<jfloatArray>(env->GetObjectArrayElement(inputData, i));
        jfloat *datas = env->GetFloatArrayElements(curArray, JNI_FALSE);
        std::string curTensorName = input_names_ptr[i];
        std::shared_ptr<Tensor> cur_input_tensor = inMap[curTensorName];
        jint dataNum = env->GetArrayLength(curArray);
        TensorDesc tensorDesc = cur_input_tensor->get_desc();
        mem_ptr[i] = ((CpuMemory *)(cur_input_tensor->get_memory()))->get_ptr();
        transformFromFloat(tensorDesc.dt, datas, mem_ptr[i], dataNum);
        env->ReleaseFloatArrayElements(curArray, datas, 0);
        env->DeleteLocalRef(curArray);
    }

    RunModel(ih, ir, num_input, (const char **)input_names_ptr, mem_ptr);
    for (int i = 0; i < num_input; i++) {
        free(input_names_ptr[i]);
    }
    free(input_names_ptr);
    free(mem_ptr);
}

std::string getString(JNIEnv *env, jstring jstring1)
{
    const char *path = NULL;
    path = env->GetStringUTFChars(jstring1, 0);
    std::string spFn(path);
    env->ReleaseStringUTFChars(jstring1, path);
    return spFn;
}

extern "C" JNIEXPORT jobject JNICALL BOLT_JNI_PREFIX(BoltModel_getOutput)(
    JNIEnv *env, jobject, jlong ResultHandleAddr, jstring boltResultPath)
{
    std::string boltResultClassPath = getString(env, boltResultPath);
    jclass stucls = env->FindClass(boltResultClassPath.c_str());

    jmethodID constrocMID =
        env->GetMethodID(stucls, "<init>", "([[F[[I[Ljava/lang/String;[Ljava/lang/String;)V");

    ResultHandleInner *ir_inner = (ResultHandleInner *)ResultHandleAddr;
    DataDesc *outputArrPtr = (*ir_inner).outputArr;
    int num_outputs = (*ir_inner).num_outputs;

    jobjectArray output_values;
    jclass floatArrCls = env->FindClass("[F");
    output_values = env->NewObjectArray(num_outputs, floatArrCls, nullptr);
    jobjectArray output_dimension;
    jclass intArrCls = env->FindClass("[I");
    output_dimension = env->NewObjectArray(num_outputs, intArrCls, nullptr);

    jobjectArray output_names_arr;
    output_names_arr = (jobjectArray)env->NewObjectArray(
        num_outputs, env->FindClass("java/lang/String"), env->NewStringUTF(""));

    jobjectArray df_arr;
    df_arr = (jobjectArray)env->NewObjectArray(
        num_outputs, env->FindClass("java/lang/String"), env->NewStringUTF(""));

    for (int i = 0; i < num_outputs; i++) {
        std::string cur_output_name = outputArrPtr[i].name;
        env->SetObjectArrayElement(output_names_arr, i, env->NewStringUTF(cur_output_name.c_str()));
        DataType cur_data_type = outputArrPtr[i].dt;
        DataFormat cur_data_format = outputArrPtr[i].df;
        std::string cur_data_format_str = DataFormat2str(cur_data_format);
        env->SetObjectArrayElement(df_arr, i, env->NewStringUTF(cur_data_format_str.c_str()));

        void *cur_dataPtr = outputArrPtr[i].dataPtr;
        int tensorNumber = calculateLength((int *)outputArrPtr[i].dims, 4);
        jfloatArray floatArr = env->NewFloatArray(tensorNumber);
        float *tmp_output_values = env->GetFloatArrayElements(floatArr, NULL);

        jint tmp_output_dimensions[4];
        jintArray intArr = env->NewIntArray(4);

        for (int j = 0; j < 4; j++) {
            tmp_output_dimensions[j] = (int)(outputArrPtr[i].dims[j]);
        }

        transformToFloat(cur_data_type, cur_dataPtr, tmp_output_values, tensorNumber);
        env->SetFloatArrayRegion(floatArr, 0, tensorNumber, tmp_output_values);
        env->SetObjectArrayElement(output_values, i, floatArr);
        env->ReleaseFloatArrayElements(floatArr, tmp_output_values, 0);

        env->DeleteLocalRef(floatArr);

        env->SetIntArrayRegion(intArr, 0, 4, tmp_output_dimensions);
        env->SetObjectArrayElement(output_dimension, i, intArr);
        env->DeleteLocalRef(intArr);
    }

    jobject bolt_result_obj = env->NewObject(
        stucls, constrocMID, output_values, output_dimension, output_names_arr, df_arr);
    env->DeleteLocalRef(stucls);
    env->DeleteLocalRef(intArrCls);
    env->DeleteLocalRef(output_values);
    env->DeleteLocalRef(output_dimension);
    env->DeleteLocalRef(output_names_arr);
    env->DeleteLocalRef(df_arr);
    return bolt_result_obj;
}

extern "C" JNIEXPORT jlong JNICALL BOLT_JNI_PREFIX(BoltModel_cloneResultHandle)(
    JNIEnv *, jobject, jlong ResultHandleAddr)
{
    ResultHandle ir = (ResultHandle)ResultHandleAddr;
    return (long)CloneResultHandle(ir);
}

extern "C" JNIEXPORT void JNICALL BOLT_JNI_PREFIX(BoltModel_freeResultHandle)(
    JNIEnv *, jobject, jlong ResultHandleAddr)
{
    ResultHandle ir = (ResultHandle)ResultHandleAddr;
    FreeResultHandle(ir);
}

extern "C" JNIEXPORT void JNICALL BOLT_JNI_PREFIX(BoltModel_destroyModel)(
    JNIEnv *, jobject, jlong modelAddr)
{
    ModelHandle ih = (ModelHandle)modelAddr;
    DestroyModel(ih);
}
#endif
