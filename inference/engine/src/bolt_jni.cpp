// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifdef _USE_API_JAVA
#include "BoltModel.h"
#include "cnn.h"
#include "../api/c/bolt.h"
#include "profiling.h"

struct ModelHandleInner {
    void *ms;
    void *cnn;
    HARDWARE_TYPE device;
    void *algoPath;
    bool useFileStream;
};

typedef struct DataDesc {
    DataType dt;
    DataFormat df;
    U32 nDims = 0;
    U32 dims[DIM_LEN] = {0};
    char name[NAME_LEN] = {0};
    void *data;
} DataDesc;

typedef struct {
    U32 num_data;
    DataDesc *data;
    HARDWARE_TYPE device;
} ResultHandleInner;

AFFINITY_TYPE str2AFFINITY_TYPE(std::string affinity)
{
    static std::map<std::string, AFFINITY_TYPE> m = {
        {"CPU", CPU},
        {"CPU_HIGH_PERFORMANCE", CPU_HIGH_PERFORMANCE},
        {"CPU_LOW_POWER", CPU_LOW_POWER},
        {"GPU", GPU},
    };
    AFFINITY_TYPE ret = CPU_HIGH_PERFORMANCE;
    if (m.find(affinity) != m.end()) {
        ret = m[affinity];
    } else {
        UNI_ERROR_LOG("unsupported JNI CPU affinity setting %s.\n", affinity.c_str());
    }
    return ret;
}

HARDWARE_TYPE str2HARDWARE_TYPE(std::string device)
{
    static std::map<std::string, HARDWARE_TYPE> m = {
        {"CPU_ARM_V7", CPU_ARM_V7},
        {"CPU_ARM_V8", CPU_ARM_V8},
        {"CPU_ARM_A55", CPU_ARM_A55},
        {"CPU_ARM_A76", CPU_ARM_A76},
        {"GPU_MALI", GPU_MALI},
        {"GPU_QUALCOMM", GPU_QUALCOMM},
        {"CPU_X86_AVX2", CPU_X86_AVX2},
        {"CPU_X86_AVX512", CPU_X86_AVX512},
        {"CPU_SERIAL", CPU_SERIAL},
    };
    HARDWARE_TYPE ret = CPU_ARM_V8;
    if (m.find(device) != m.end()) {
        ret = m[device];
    } else {
        UNI_ERROR_LOG("unsupported JNI device setting %s.\n", device.c_str());
    }
    return ret;
}

DATA_TYPE str2DATA_TYPE(std::string dt)
{
    static std::map<std::string, DATA_TYPE> m = {
        {"FP32", FP_32},
        {"FP16", FP_16},
        {"INT32", INT_32},
        {"UINT32", UINT_32},
        {"INT8", INT_8},
        {"UINT8", UINT_8},
    };
    DATA_TYPE ret = FP_32;
    if (m.find(dt) != m.end()) {
        ret = m[dt];
    } else {
        UNI_ERROR_LOG("unsupported JNI data type setting %s.\n", dt.c_str());
    }
    return ret;
}

DATA_FORMAT str2DATA_FORMAT(std::string df)
{
    static std::map<std::string, DATA_FORMAT> m = {
        {"NCHW", NCHW},
        {"NHWC", NHWC},
        {"NCHWC8", NCHWC8},
        {"NCHWC4", NCHWC4},
        {"MTK", MTK},
        {"NORMAL", NORMAL},
        {"SCALAR", SCALAR},
    };
    DATA_FORMAT ret = NCHW;
    if (m.find(df) != m.end()) {
        ret = m[df];
    } else {
        UNI_ERROR_LOG("unsupported JNI data format setting %s.\n", df.c_str());
    }
    return ret;
}

std::string DataFormat2str(DataFormat df)
{
    static std::map<DataFormat, std::string> m = {
        {DF_NCHW, "NCHW"},
        {DF_NHWC, "NHWC"},
        {DF_NCHWC8, "NCHWC8"},
        {DF_NCHWC4, "NCHWC4"},
        {DF_MTK, "MTK"},
        {DF_NORMAL, "NORMAL"},
        {DF_SCALAR, "SCALAR"},
    };
    std::string ret = "NCHW";
    if (m.find(df) != m.end()) {
        ret = m[df];
    } else {
        UNI_ERROR_LOG("JNI can not process inner DataFormat %s.\n", DataFormatName()[df]);
    }
    return ret;
}

std::string getString(JNIEnv *env, jstring str)
{
    const char *p = NULL;
    p = env->GetStringUTFChars(str, 0);
    std::string ret(p);
    env->ReleaseStringUTFChars(str, p);
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
    UNI_PROFILE(
        {
            if (env->GetArrayLength(input_names) != num) {
                UNI_ERROR_LOG("input name array length %d is not equal to input num %d\n",
                    env->GetArrayLength(input_names), num);
            }
            if (env->GetArrayLength(n) != num) {
                UNI_ERROR_LOG("input N array length %d is not equal to input num %d\n",
                    env->GetArrayLength(n), num);
            }
            if (env->GetArrayLength(c) != num) {
                UNI_ERROR_LOG("input C array length %d is not equal to input num %d\n",
                    env->GetArrayLength(c), num);
            }
            if (env->GetArrayLength(h) != num) {
                UNI_ERROR_LOG("input H array length %d is not equal to input num %d\n",
                    env->GetArrayLength(h), num);
            }
            if (env->GetArrayLength(w) != num) {
                UNI_ERROR_LOG("input W array length %d is not equal to input num %d\n",
                    env->GetArrayLength(w), num);
            }
            if (env->GetArrayLength(dt_input) != num) {
                UNI_ERROR_LOG("input DataType array length %d is not equal to input num %d\n",
                    env->GetArrayLength(dt_input), num);
            }
            if (env->GetArrayLength(df_input) != num) {
                UNI_ERROR_LOG("input DataFormat array length %d is not equal to input num %d\n",
                    env->GetArrayLength(df_input), num);
            }
            int *data_n = (int *)UNI_MALLOC(num * sizeof(int));
            int *data_c = (int *)UNI_MALLOC(num * sizeof(int));
            int *data_h = (int *)UNI_MALLOC(num * sizeof(int));
            int *data_w = (int *)UNI_MALLOC(num * sizeof(int));
            char **data_name = (char **)UNI_MALLOC(num * sizeof(char *));
            DATA_TYPE *data_dt = (DATA_TYPE *)UNI_MALLOC(num * sizeof(DATA_TYPE));
            DATA_FORMAT *data_df = (DATA_FORMAT *)UNI_MALLOC(num * sizeof(DATA_FORMAT));
            jint *curArray_n = env->GetIntArrayElements(n, 0);
            jint *curArray_c = env->GetIntArrayElements(c, 0);
            jint *curArray_h = env->GetIntArrayElements(h, 0);
            jint *curArray_w = env->GetIntArrayElements(w, 0);
            for (int i = 0; i < num; i++) {
                data_n[i] = curArray_n[i];
                data_c[i] = curArray_c[i];
                data_h[i] = curArray_h[i];
                data_w[i] = curArray_w[i];

                std::string name = getString(
                    env, static_cast<jstring>(env->GetObjectArrayElement(input_names, i)));
                data_name[i] = (char *)UNI_MALLOC(sizeof(char) * (name.length() + 1));
                UNI_STRCPY(data_name[i], name.c_str());

                data_dt[i] = str2DATA_TYPE(
                    getString(env, static_cast<jstring>(env->GetObjectArrayElement(dt_input, i))));
                data_df[i] = str2DATA_FORMAT(
                    getString(env, static_cast<jstring>(env->GetObjectArrayElement(df_input, i))));
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
        },
        std::string(__FUNCTION__), std::string("JNI::get_parameter"));
}

template <typename T>
unsigned int calculateLength(T *array, int num)
{
    unsigned int length = 0;
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
    UNI_DEBUG_LOG("JNI %s...\n", __FUNCTION__);
    ModelHandle ret;
    UNI_PROFILE(
        {
            std::string p0 = getString(env, modelPath);
            AFFINITY_TYPE p1 = str2AFFINITY_TYPE(getString(env, affinity));
            ret = CreateModel(p0.c_str(), p1, NULL);
            if (nullptr == (ModelHandleInner *)ret) {
                UNI_ERROR_LOG("Bolt instance not created\n");
                ret = 0;
            }
        },
        std::string(__FUNCTION__), std::string("JNI::create_model"));
    UNI_DEBUG_LOG("JNI %s end.\n", __FUNCTION__);
    return (long long)ret;
}

extern "C" JNIEXPORT jlong JNICALL BOLT_JNI_PREFIX(BoltModel_cloneModel)(
    JNIEnv *env, jobject, jlong modelAddr)
{
    UNI_DEBUG_LOG("JNI %s...\n", __FUNCTION__);
    ModelHandle ret;
    UNI_PROFILE({ ret = CloneModel((ModelHandle)modelAddr); }, std::string(__FUNCTION__),
        std::string("JNI::clone_model"));
    UNI_DEBUG_LOG("JNI %s end.\n", __FUNCTION__);
    return (long long)ret;
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
    UNI_DEBUG_LOG("JNI %s...\n", __FUNCTION__);
    UNI_PROFILE(
        {
            ModelHandle ih = (ModelHandle)modelAddr;
            char **data_name = nullptr;
            int *data_n = nullptr;
            int *data_c = nullptr;
            int *data_h = nullptr;
            int *data_w = nullptr;
            DATA_TYPE *data_dt = nullptr;
            DATA_FORMAT *data_df = nullptr;
            getInputParameters(env, num_input, input_names, &data_name, n, &data_n, c, &data_c, h,
                &data_h, w, &data_w, dt_input, &data_dt, df_input, &data_df);

            PrepareModel(ih, num_input, (const char **)data_name, data_n, data_c, data_h, data_w,
                data_dt, data_df);

            UNI_FREE(data_n);
            UNI_FREE(data_c);
            UNI_FREE(data_h);
            UNI_FREE(data_w);
            for (int i = 0; i < num_input; i++) {
                UNI_FREE(data_name[i]);
            }
            UNI_FREE(data_name);
            UNI_FREE(data_dt);
            UNI_FREE(data_df);
        },
        std::string(__FUNCTION__), std::string("JNI::prepare_model"));
    UNI_DEBUG_LOG("JNI %s end.\n", __FUNCTION__);
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
    UNI_DEBUG_LOG("JNI %s...\n", __FUNCTION__);
    UNI_PROFILE(
        {
            ModelHandle ih = (ModelHandle)modelAddr;
            char **data_name = nullptr;
            int *data_n = nullptr;
            int *data_c = nullptr;
            int *data_h = nullptr;
            int *data_w = nullptr;
            DATA_TYPE *data_dt = nullptr;
            DATA_FORMAT *data_df = nullptr;
            getInputParameters(env, num_input, input_names, &data_name, n, &data_n, c, &data_c, h,
                &data_h, w, &data_w, dt_input, &data_dt, df_input, &data_df);

            ResizeModelInput(ih, num_input, (const char **)data_name, data_n, data_c, data_h,
                data_w, data_dt, data_df);

            UNI_FREE(data_n);
            UNI_FREE(data_c);
            UNI_FREE(data_h);
            UNI_FREE(data_w);
            for (int i = 0; i < num_input; i++) {
                UNI_FREE(data_name[i]);
            }
            UNI_FREE(data_name);
            UNI_FREE(data_dt);
            UNI_FREE(data_df);
        },
        std::string(__FUNCTION__), std::string("JNI::resize_input"));
    UNI_DEBUG_LOG("JNI %s end.\n", __FUNCTION__);
}

extern "C" JNIEXPORT jlong JNICALL BOLT_JNI_PREFIX(BoltModel_allocAllResultHandle)(
    JNIEnv *, jobject, jlong modelAddr)
{
    UNI_DEBUG_LOG("JNI %s...\n", __FUNCTION__);
    ResultHandle ir;
    UNI_PROFILE({ ir = AllocAllResultHandle((ModelHandle)modelAddr); }, std::string(__FUNCTION__),
        std::string("JNI::alloc_all_result"));
    UNI_DEBUG_LOG("JNI %s end.\n", __FUNCTION__);
    return (long long)ir;
}

extern "C" JNIEXPORT jlong JNICALL BOLT_JNI_PREFIX(BoltModel_allocSpecificResultHandle)(
    JNIEnv *env, jobject, jlong modelAddr, jint num_outputs, jobjectArray outputNames)
{
    UNI_DEBUG_LOG("JNI %s...\n", __FUNCTION__);
    ResultHandle ir;
    UNI_PROFILE(
        {
            if (env->GetArrayLength(outputNames) != num_outputs) {
                UNI_ERROR_LOG("output name array length %d is not equal to output num %d\n",
                    env->GetArrayLength(outputNames), num_outputs);
            }
            ModelHandle ih = (ModelHandle)modelAddr;
            char **names = (char **)UNI_MALLOC(sizeof(char *) * num_outputs);
            for (int i = 0; i < num_outputs; i++) {
                std::string name = getString(
                    env, static_cast<jstring>(env->GetObjectArrayElement(outputNames, i)));
                names[i] = (char *)UNI_MALLOC(sizeof(char) * (name.length() + 1));
                UNI_STRCPY(names[i], name.c_str());
            }
            ir = AllocSpecificResultHandle(ih, num_outputs, (const char **)names);
            for (int i = 0; i < num_outputs; i++) {
                UNI_FREE(names[i]);
            }
            UNI_FREE(names);
        },
        std::string(__FUNCTION__), std::string("JNI::alloc_result"));
    UNI_DEBUG_LOG("JNI %s end.\n", __FUNCTION__);
    return (long long)ir;
}

extern "C" JNIEXPORT void JNICALL BOLT_JNI_PREFIX(BoltModel_setRuntimeDeviceJNI)(
    JNIEnv *env, jobject, jlong modelAddr, jint cpu_id, jstring device)
{
    UNI_DEBUG_LOG("JNI %s...\n", __FUNCTION__);
    ModelHandle ih = (ModelHandle)modelAddr;
    HARDWARE_TYPE p = str2HARDWARE_TYPE(getString(env, device));
    SetRuntimeDevice(ih, cpu_id, p);
    UNI_DEBUG_LOG("JNI %s end.\n", __FUNCTION__);
}

extern "C" JNIEXPORT void JNICALL BOLT_JNI_PREFIX(BoltModel_setRuntimeDeviceDynamicJNI)(
    JNIEnv *env, jobject, jlong modelAddr)
{
    UNI_DEBUG_LOG("JNI %s...\n", __FUNCTION__);
    ModelHandle ih = (ModelHandle)modelAddr;
    SetRuntimeDeviceDynamic(ih);
    UNI_DEBUG_LOG("JNI %s end.\n", __FUNCTION__);
}

extern "C" JNIEXPORT void JNICALL BOLT_JNI_PREFIX(BoltModel_setNumThreads)(
    JNIEnv *env, jobject, jint threads)
{
    UNI_DEBUG_LOG("JNI %s...\n", __FUNCTION__);
    SetNumThreads(threads);
    UNI_DEBUG_LOG("JNI %s end.\n", __FUNCTION__);
}

extern "C" JNIEXPORT void JNICALL BOLT_JNI_PREFIX(BoltModel_runModel)(JNIEnv *env,
    jobject,
    jlong modelAddr,
    jlong ResultHandleAddr,
    jint num_input,
    jobjectArray input_names,
    jobjectArray inputData)
{
    UNI_DEBUG_LOG("JNI %s...\n", __FUNCTION__);
    UNI_PROFILE(
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

            char **names = (char **)UNI_MALLOC(sizeof(char *) * num_input);
            std::vector<jfloatArray> cache0(num_input);
            std::vector<jfloat *> datas(num_input);
            for (int i = 0; i < num_input; i++) {
                std::string name = getString(
                    env, static_cast<jstring>(env->GetObjectArrayElement(input_names, i)));
                names[i] = (char *)UNI_MALLOC(sizeof(char) * (name.length() + 1));
                UNI_STRCPY(names[i], name.c_str());

                jfloatArray array =
                    static_cast<jfloatArray>(env->GetObjectArrayElement(inputData, i));
                jfloat *p = env->GetFloatArrayElements(array, NULL);
                cache0[i] = array;
                datas[i] = p;
            }
            std::vector<DATA_TYPE> dts(num_input, FP_32);
            RunModelWithType(
                ih, ir, num_input, (const char **)names, dts.data(), (void **)datas.data(), nullptr);
            for (int i = 0; i < num_input; i++) {
                UNI_FREE(names[i]);
                env->ReleaseFloatArrayElements(cache0[i], datas[i], 0);
            }
            UNI_FREE(names);
        },
        std::string(__FUNCTION__), std::string("JNI::run"));
    UNI_DEBUG_LOG("JNI %s end.\n", __FUNCTION__);
}

extern "C" JNIEXPORT jobject JNICALL BOLT_JNI_PREFIX(BoltModel_getOutput)(
    JNIEnv *env, jobject, jlong ResultHandleAddr, jstring boltResultPath)
{
    UNI_DEBUG_LOG("JNI %s...\n", __FUNCTION__);
    jobject ret;
    UNI_PROFILE(
        {
            std::string boltResultClassPath = getString(env, boltResultPath);
            jclass stucls = env->FindClass(boltResultClassPath.c_str());
            jmethodID constrocMID = env->GetMethodID(
                stucls, "<init>", "([[F[[I[Ljava/lang/String;[Ljava/lang/String;)V");

            ResultHandleInner *ir_inner = (ResultHandleInner *)ResultHandleAddr;
            DataDesc *p = ir_inner->data;
            int num_outputs = ir_inner->num_data;

            jobjectArray value = env->NewObjectArray(num_outputs, env->FindClass("[F"), nullptr);
            jobjectArray dimension = env->NewObjectArray(num_outputs, env->FindClass("[I"), nullptr);
            jobjectArray name = env->NewObjectArray(
                num_outputs, env->FindClass("java/lang/String"), env->NewStringUTF(""));
            jobjectArray dataFormat = env->NewObjectArray(
                num_outputs, env->FindClass("java/lang/String"), env->NewStringUTF(""));
            for (int i = 0; i < num_outputs; i++) {
                env->SetObjectArrayElement(name, i, env->NewStringUTF(p[i].name));
                DataFormat df = p[i].df;
                env->SetObjectArrayElement(
                    dataFormat, i, env->NewStringUTF(DataFormat2str(df).c_str()));

                unsigned int length = calculateLength(p[i].dims, p[i].nDims);
                jfloatArray p0 = env->NewFloatArray(length);
                if (p[i].df == DF_NCHWC8 || p[i].df == DF_NCHWC4) {
                }
                if (p[i].dt == DT_F32) {
                    env->SetFloatArrayRegion(p0, 0, length, (const jfloat *)p[i].data);
                } else {
                    float *p00 = env->GetFloatArrayElements(p0, NULL);
                    transformToFloat(p[i].dt, p[i].data, p00, length);
                    env->ReleaseFloatArrayElements(p0, p00, 0);
                }
                env->SetObjectArrayElement(value, i, p0);
                env->DeleteLocalRef(p0);

                jintArray p1 = env->NewIntArray(p[i].nDims);
                jint *p10 = env->GetIntArrayElements(p1, NULL);
                for (U32 j = 0; j < p[i].nDims; j++) {
                    p10[j] = p[i].dims[j];
                }
                env->ReleaseIntArrayElements(p1, p10, 0);
                env->SetObjectArrayElement(dimension, i, p1);
                env->DeleteLocalRef(p1);
            }

            ret = env->NewObject(stucls, constrocMID, value, dimension, name, dataFormat);
            env->DeleteLocalRef(stucls);
            env->DeleteLocalRef(value);
            env->DeleteLocalRef(dimension);
            env->DeleteLocalRef(name);
            env->DeleteLocalRef(dataFormat);
        },
        std::string(__FUNCTION__), std::string("JNI::get_output"));
    UNI_DEBUG_LOG("JNI %s end.\n", __FUNCTION__);
    return ret;
}

extern "C" JNIEXPORT jlong JNICALL BOLT_JNI_PREFIX(BoltModel_cloneResultHandle)(
    JNIEnv *, jobject, jlong ResultHandleAddr)
{
    UNI_DEBUG_LOG("JNI %s...\n", __FUNCTION__);
    ResultHandle ret;
    UNI_PROFILE({ ret = CloneResultHandle((ResultHandle)ResultHandleAddr); },
        std::string(__FUNCTION__), std::string("JNI::clone_output"));
    UNI_DEBUG_LOG("JNI %s end.\n", __FUNCTION__);
    return (long long)ret;
}

extern "C" JNIEXPORT void JNICALL BOLT_JNI_PREFIX(BoltModel_freeResultHandle)(
    JNIEnv *, jobject, jlong ResultHandleAddr)
{
    UNI_DEBUG_LOG("JNI %s...\n", __FUNCTION__);
    UNI_PROFILE({ FreeResultHandle((ResultHandle)ResultHandleAddr); }, std::string(__FUNCTION__),
        std::string("JNI::free_output"));
    UNI_DEBUG_LOG("JNI %s end.\n", __FUNCTION__);
}

extern "C" JNIEXPORT void JNICALL BOLT_JNI_PREFIX(BoltModel_destroyModel)(
    JNIEnv *, jobject, jlong modelAddr)
{
    UNI_DEBUG_LOG("JNI %s...\n", __FUNCTION__);
    UNI_PROFILE({ DestroyModel((ModelHandle)modelAddr); }, std::string(__FUNCTION__),
        std::string("JNI::free_model"));
    UNI_DEBUG_LOG("JNI %s end.\n", __FUNCTION__);
}
#endif
