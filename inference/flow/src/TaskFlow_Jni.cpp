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

#include "TaskFlow.h"
#include "flow.h"

AffinityPolicy str2AffinityPolicy(std::string affinity_str)
{
    AffinityPolicy ret = AFFINITY_CPU_HIGH_PERFORMANCE;
    if (affinity_str == "CPU_HIGH_PERFORMANCE") {
        ret = AFFINITY_CPU_HIGH_PERFORMANCE;
    } else if (affinity_str == "CPU_LOW_POWER") {
        ret = AFFINITY_CPU_LOW_POWER;
    } else if (affinity_str == "GPU") {
        ret = AFFINITY_GPU;
    } else {
        UNI_ERROR_LOG("unsupported JNI CPU affinity setting %s\n", affinity_str.c_str());
    }
    return ret;
}

DataType str2DataType(std::string data_type)
{
    DataType ret = DT_F32;
    if (data_type == "FP32") {
        ret = DT_F32;
#ifdef __aarch64__
    } else if (data_type == "FP16") {
        ret = DT_F16;
#endif
    } else if (data_type == "INT32") {
        ret = DT_I32;
    } else if (data_type == "UINT32") {
        ret = DT_U32;
    } else {
        UNI_ERROR_LOG("unsupported JNI data type setting %s\n", data_type.c_str());
    }
    return ret;
}

DataFormat str2DataFormat(std::string data_format)
{
    DataFormat ret = DF_NCHW;
    if (data_format == "NCHW") {
        ret = DF_NCHW;
    } else if (data_format == "NHWC") {
        ret = DF_NHWC;
    } else if (data_format == "MTK") {
        ret = DF_MTK;
    } else if (data_format == "NORMAL") {
        ret = DF_NORMAL;
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

JNIEXPORT void JNICALL BOLT_JNI_PREFIX(TaskFlow_taskEnqueue)(
    JNIEnv *, jobject, jlong flow_addr, jlong task_addr)
{
    Flow *flow = (Flow *)flow_addr;
    Task *task = (Task *)task_addr;
    flow->enqueue(*task);
}

JNIEXPORT jlongArray JNICALL BOLT_JNI_PREFIX(TaskFlow_tasksDequeue)(
    JNIEnv *env, jobject, jlong flow_addr, jboolean block)
{
    Flow *flow = (Flow *)flow_addr;
    std::vector<Task> results = flow->dequeue(block);
    int length = results.size();
    std::vector<jlong> tasks_addr(length);
    jlongArray newArr = env->NewLongArray(length);
    for (int i = 0; i < length; i++) {
        Task *task = new Task(&results[i]);
        tasks_addr[i] = (jlong)task;
    }
    env->SetLongArrayRegion(newArr, 0, length, tasks_addr.data());
    return newArr;
}

JNIEXPORT jobject JNICALL BOLT_JNI_PREFIX(TaskFlow_getTaskResult)(JNIEnv *env,
    jobject,
    jlong task_addr,
    jint outputNumber,
    jobjectArray outputNames,
    jstring boltResultPath)
{
    const char *boltResultClassPath = env->GetStringUTFChars(boltResultPath, 0);
    jclass stucls = env->FindClass(boltResultClassPath);
    env->ReleaseStringUTFChars(boltResultPath, boltResultClassPath);
    jmethodID constrocMID =
        env->GetMethodID(stucls, "<init>", "([[F[[I[Ljava/lang/String;[Ljava/lang/String;)V");

    jobjectArray output_values;
    jclass floatArrCls = env->FindClass("[F");
    output_values = env->NewObjectArray(outputNumber, floatArrCls, nullptr);

    jobjectArray output_dimension;
    jclass intArrCls = env->FindClass("[I");
    output_dimension = env->NewObjectArray(outputNumber, intArrCls, nullptr);

    jobjectArray output_names_arr;
    output_names_arr = (jobjectArray)env->NewObjectArray(
        outputNumber, env->FindClass("java/lang/String"), env->NewStringUTF(""));

    jobjectArray df_arr;
    df_arr = (jobjectArray)env->NewObjectArray(
        outputNumber, env->FindClass("java/lang/String"), env->NewStringUTF(""));

    Task *task = (Task *)task_addr;
    for (int i = 0; i < outputNumber; i++) {
        //output name
        jstring output_name_str = (jstring)(env->GetObjectArrayElement(outputNames, i));
        const char *output_name_str_ptr = env->GetStringUTFChars(output_name_str, 0);
        std::string output_name = output_name_str_ptr;
        env->SetObjectArrayElement(output_names_arr, i, env->NewStringUTF(output_name.c_str()));
        env->ReleaseStringUTFChars(output_name_str, output_name_str_ptr);
        env->DeleteLocalRef(output_name_str);

        //output data format
        std::shared_ptr<Tensor> cur_output_tensor = task->data[output_name];
        TensorDesc tensorDesc = cur_output_tensor->get_desc();
        std::string cur_data_format_str = DataFormat2str(tensorDesc.df);
        env->SetObjectArrayElement(df_arr, i, env->NewStringUTF(cur_data_format_str.c_str()));

        //output data mimensions
        jint tmp_output_dimensions[4];
        jintArray intArr = env->NewIntArray(4);
        for (int j = 0; j < 4; j++) {
            tmp_output_dimensions[j] = tensorDesc.dims[j];
        }
        env->SetIntArrayRegion(intArr, 0, 4, tmp_output_dimensions);
        env->SetObjectArrayElement(output_dimension, i, intArr);
        env->DeleteLocalRef(intArr);

        //output data values
        void *cur_dataPtr = ((CpuMemory *)(cur_output_tensor->get_memory()))->get_ptr();
        int data_Length = calculateLength((int *)tensorDesc.dims, 4);
        jfloatArray floatArr = env->NewFloatArray(data_Length);
        float *tmp_output_values = env->GetFloatArrayElements(floatArr, NULL);
        transformToFloat(tensorDesc.dt, cur_dataPtr, tmp_output_values, data_Length);
        env->SetFloatArrayRegion(floatArr, 0, data_Length, tmp_output_values);
        env->SetObjectArrayElement(output_values, i, floatArr);
        env->ReleaseFloatArrayElements(floatArr, tmp_output_values, 0);
        env->DeleteLocalRef(floatArr);
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

JNIEXPORT jlong JNICALL BOLT_JNI_PREFIX(TaskFlow_createFlow)(JNIEnv *env,
    jobject,
    jstring graphPath,
    jstring precision,
    jstring affinityPolicy,
    jint cpuThreads,
    jboolean useGPU)
{
    const char *garphPathPtr = env->GetStringUTFChars(graphPath, JNI_FALSE);
    const char *precisionPtr = env->GetStringUTFChars(precision, JNI_FALSE);
    const char *affinityPolicyPtr = env->GetStringUTFChars(affinityPolicy, JNI_FALSE);
    DataType dataType = str2DataType(precisionPtr);
    AffinityPolicy affinity = str2AffinityPolicy(affinityPolicyPtr);
    std::vector<std::string> graphPathVec = {garphPathPtr};
    Flow *flow = new Flow();
    flow->init(graphPathVec, dataType, affinity, cpuThreads, useGPU);
    return (jlong)(flow);
}

JNIEXPORT jlong JNICALL BOLT_JNI_PREFIX(TaskFlow_createTask)(JNIEnv *env,
    jobject,
    jstring graphPath,
    jint inputNumber,
    jintArray inputN,
    jintArray inputC,
    jintArray inputH,
    jintArray inputW,
    jobjectArray inputNames,
    jobjectArray inputDataType,
    jobjectArray inputDataFormat,
    jobjectArray inputData,
    jint outputNumber,
    jintArray outputN,
    jintArray outputC,
    jintArray outputH,
    jintArray outputW,
    jobjectArray outputNames,
    jobjectArray outputDataType,
    jobjectArray outputDataFormat)
{
    std::map<std::string, std::shared_ptr<Tensor>> tensors;
    jint *input_n = env->GetIntArrayElements(inputN, 0);
    jint *input_c = env->GetIntArrayElements(inputC, 0);
    jint *input_h = env->GetIntArrayElements(inputH, 0);
    jint *input_w = env->GetIntArrayElements(inputW, 0);
    for (int i = 0; i < inputNumber; i++) {
        jstring input_name_str = (jstring)(env->GetObjectArrayElement(inputNames, i));
        jstring input_data_type_str = (jstring)(env->GetObjectArrayElement(inputDataType, i));
        jstring input_data_format_str = (jstring)(env->GetObjectArrayElement(inputDataFormat, i));
        const char *input_name_str_ptr = env->GetStringUTFChars(input_name_str, 0);
        const char *input_data_type_str_ptr = env->GetStringUTFChars(input_data_type_str, 0);
        const char *input_data_format_str_ptr = env->GetStringUTFChars(input_data_format_str, 0);

        std::string input_name = input_name_str_ptr;
        DataType data_type = str2DataType(input_data_type_str_ptr);
        DataFormat data_format = str2DataFormat(input_data_format_str_ptr);
        env->ReleaseStringUTFChars(input_name_str, input_name_str_ptr);
        env->DeleteLocalRef(input_name_str);
        env->ReleaseStringUTFChars(input_data_type_str, input_data_type_str_ptr);
        env->DeleteLocalRef(input_data_type_str);
        env->ReleaseStringUTFChars(input_data_format_str, input_data_format_str_ptr);
        env->DeleteLocalRef(input_data_format_str);

        //input data values
        jfloatArray curArray = static_cast<jfloatArray>(env->GetObjectArrayElement(inputData, i));
        jfloat *datas = env->GetFloatArrayElements(curArray, JNI_FALSE);
        jint dataNum = env->GetArrayLength(curArray);
        tensors[input_name] = std::shared_ptr<Tensor>(new Tensor());
        TensorDesc inputDesc;
        if (data_format == DF_NORMAL) {
            inputDesc = tensor2df(data_type, data_format, input_n[i], input_c[i]);
        } else if (data_format == DF_MTK) {
            inputDesc = tensor3df(data_type, data_format, input_n[i], input_c[i], input_h[i]);
        } else {
            UNI_ERROR_LOG("unsupported JNI input data format %d\n", data_format);
            return 0;
        }
        tensors[input_name]->resize(inputDesc);
        tensors[input_name]->alloc();
        auto ptr = ((CpuMemory *)(tensors[input_name]->get_memory()))->get_ptr();
        transformFromFloat(inputDesc.dt, datas, ptr, dataNum);

        env->ReleaseFloatArrayElements(curArray, datas, 0);
        env->DeleteLocalRef(curArray);
    }
    env->ReleaseIntArrayElements(inputN, input_n, 0);
    env->ReleaseIntArrayElements(inputC, input_c, 0);
    env->ReleaseIntArrayElements(inputH, input_h, 0);
    env->ReleaseIntArrayElements(inputW, input_w, 0);

    //output tensors
    jint *output_n = env->GetIntArrayElements(outputN, 0);
    jint *output_c = env->GetIntArrayElements(outputC, 0);
    jint *output_h = env->GetIntArrayElements(outputH, 0);
    jint *output_w = env->GetIntArrayElements(outputW, 0);
    for (int i = 0; i < outputNumber; i++) {
        jstring output_name_str = (jstring)(env->GetObjectArrayElement(outputNames, i));
        jstring output_data_type_str = (jstring)(env->GetObjectArrayElement(outputDataType, i));
        jstring output_data_format_str = (jstring)(env->GetObjectArrayElement(outputDataFormat, i));
        const char *output_name_str_ptr = env->GetStringUTFChars(output_name_str, 0);
        const char *output_data_type_str_ptr = env->GetStringUTFChars(output_data_type_str, 0);
        const char *output_data_format_str_ptr = env->GetStringUTFChars(output_data_format_str, 0);

        std::string output_name = output_name_str_ptr;
        DataType data_type = str2DataType(output_data_type_str_ptr);
        DataFormat data_format = str2DataFormat(output_data_format_str_ptr);
        env->ReleaseStringUTFChars(output_name_str, output_name_str_ptr);
        env->DeleteLocalRef(output_name_str);
        env->ReleaseStringUTFChars(output_data_type_str, output_data_type_str_ptr);
        env->DeleteLocalRef(output_data_type_str);
        env->ReleaseStringUTFChars(output_data_format_str, output_data_format_str_ptr);
        env->DeleteLocalRef(output_data_format_str);

        //output tensors
        tensors[output_name] = std::shared_ptr<Tensor>(new Tensor());
        TensorDesc outputDesc;
        if (data_format == DF_NORMAL) {
            outputDesc = tensor2df(data_type, data_format, output_n[i], output_c[i]);
        } else if (data_format == DF_MTK) {
            outputDesc = tensor3df(data_type, data_format, output_n[i], output_c[i], output_h[i]);
        } else {
            UNI_ERROR_LOG("unsupported JNI output data format %d\n", data_format);
            return 0;
        }
        tensors[output_name]->resize(outputDesc);
        tensors[output_name]->alloc();
    }
    env->ReleaseIntArrayElements(outputN, output_n, 0);
    env->ReleaseIntArrayElements(outputC, output_c, 0);
    env->ReleaseIntArrayElements(outputH, output_h, 0);
    env->ReleaseIntArrayElements(outputW, output_w, 0);

    const char *graphPathPtr = env->GetStringUTFChars(graphPath, JNI_FALSE);
    Task *task = new Task(graphPathPtr, tensors);
    env->ReleaseStringUTFChars(graphPath, graphPathPtr);
    return (jlong)(task);
}

JNIEXPORT jint JNICALL BOLT_JNI_PREFIX(TaskFlow_taskFlowRegisterFunction)(
    JNIEnv *env, jobject, jstring functionName, jlong function)
{
    const char *functionNamePtr = env->GetStringUTFChars(functionName, JNI_FALSE);
    std::string functionNameStr = (std::string)functionNamePtr;
    return 0;
}

JNIEXPORT void JNICALL BOLT_JNI_PREFIX(TaskFlow_destroyFlow)(JNIEnv *, jobject, jlong flow_addr)
{
    Flow *flow = (Flow *)flow_addr;
    delete flow;
}

JNIEXPORT void JNICALL BOLT_JNI_PREFIX(TaskFlow_destroyTask)(JNIEnv *, jobject, jlong task_addr)
{
    Task *task = (Task *)task_addr;
    delete task;
}
#endif
