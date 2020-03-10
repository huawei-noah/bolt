// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.


// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifdef __clang__
#include <iostream>
#include <vector>
#include <jni.h>
#include "type.h"
#include "tensor_desc.h"
#include "cnn.hpp"
#include "BoltModel.h"
#include "../exports/c/bolt.h"

struct IHandleInfo {
    void* cnn;
    void* ms;
    DEVICE_TYPE deviceType;
};

typedef struct {
    U32 dims[4] = {0};
    char name[NAME_LEN] = {0};
    DataType dt;
    DataFormat df;
    void* dataPtr;
} DataDesc;

typedef struct {
    U32 num_outputs;
    DataDesc* outputArr;
    DEVICE_TYPE deviceType;
} IResultInner;

AFFINITY_TYPE str2AFFINITY_TYPE(std::string affinity_str) {
    AFFINITY_TYPE ret = HIGH_PERFORMANCE;
    if (affinity_str == "HIGH_PERFORMANCE") {
	ret = HIGH_PERFORMANCE;
    } else if (affinity_str == "LOW_POWER") {
	ret = LOW_POWER;
    } else {
        std::cerr << "[ERROR] unsupported JNI CPU affinity setting " << affinity_str << std::endl;
        exit(1);
    }
    return ret;
}

DEVICE_TYPE str2DEVICE_TYPE(std::string device_str) {
    DEVICE_TYPE ret = CPU;
    if (device_str == "CPU") {
	ret = CPU;
    } else if (device_str == "GPU") {
	ret = GPU;
    } else {
        std::cerr << "[ERROR] unsupported JNI device setting " << device_str << std::endl;
        exit(1);
    }
    return ret;
}

DATA_TYPE str2DATA_TYPE (std::string data_type) {
    DATA_TYPE ret = FP_32;
    if (data_type == "FP32") {
	ret = FP_32;
#ifdef _USE_FP16
    } else if (data_type == "FP16"){
	ret = FP_16;
#endif
    } else if (data_type == "INT32") {
        ret = INT_32;
    } else if (data_type == "UINT32") {
	ret = UINT_32;
    } else {
        std::cerr << "[ERROR] unsupported JNI data type setting " << data_type << std::endl;
        exit(1);
    }
    return ret;
}

DATA_FORMAT str2DATA_FORMAT (std::string data_format) {
    DATA_FORMAT ret = NCHW;
    if (data_format == "NCHW") {
	ret = NCHW;
    } else if (data_format == "NHWC") {
	ret = NHWC;
    } else if (data_format == "NORMAL") {
	ret = NORMAL;
    } else {
        std::cerr << "[ERROR] unsupported JNI data format setting " << data_format << std::endl;
        exit(1);
    }
    return ret;
}

std::string DataFormat2str (DataFormat data_format) {
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
	case DF_NORMAL:
	    ret = "NORMAL";
	    break;
        default:
            std::cerr << "[ERROR] unsupported JNI data format setting"<< std::endl;
            exit(1);
    }
    return ret;
}

void dataTypeConverterToFloat(void *src, DataType srcDataType, float *dst, int num) {
    switch (srcDataType) {
#ifdef _USE_FP16
        case DT_F16: {
            F16 *srcPtr = (F16 *)src;
            for (int i = 0; i < num; i++) {
                dst[i] = srcPtr[i];
            }
            break;
        }
#endif
        case DT_F32: {
            memcpy(dst, src, sizeof(float)*num);
            break;
        }
        case DT_U32: {
            U32 *srcPtr = (U32 *)src;
            for (int i = 0; i < num; i++) {
                dst[i] = srcPtr[i];
            }
            break;
        }
        case DT_I32: {
            I32 *srcPtr = (I32 *)src;
            for (int i = 0; i < num; i++) {
                dst[i] = srcPtr[i];
            }
            break;
        }
        default:
            std::cerr << "[ERROR] unsupported source data type in " << __func__ << std::endl;
            exit(1);
    }
}

void dataTypeConverterFromFloat(float *src, void *dst, DataType dstDataType, int num) {
    switch (dstDataType) {
#ifdef _USE_FP16
        case DT_F16: {
            F16 *dstPtr = (F16 *)dst;
            for (int i = 0; i < num; i++) {
                dstPtr[i] = (F16)src[i];
            }
            break;
        }
#endif
        case DT_F32: {
            memcpy(dst, src, sizeof(float)*num);
            break;
        }
        case DT_U32: {
            U32 *dstPtr = (U32 *)dst;
            for (int i = 0; i < num; i++) {
                dstPtr[i] = (U32)src[i];
            }
            break;
        }
        case DT_I32: {
            I32 *dstPtr = (I32 *)dst;
            for (int i = 0; i < num; i++) {
                dstPtr[i] = (I32)src[i];
            }
            break;
        }
        default:
            std::cerr << "[ERROR] unsupported source data type in " << __func__ << std::endl;
            exit(1);
    }
}

extern "C" JNIEXPORT jlong JNICALL Java_BoltModel_model_1create
  (JNIEnv *env, jobject, jstring modelPath, jstring affinity, jstring device) {
    const char* modelPathPtr = env->GetStringUTFChars(modelPath, JNI_FALSE);
    const char* affinityPtr = env->GetStringUTFChars(affinity, JNI_FALSE);
    const char* devicePtr = env->GetStringUTFChars(device, JNI_FALSE);
    std::string affinity_str = (std::string)affinityPtr;
    AFFINITY_TYPE affinity_cur = str2AFFINITY_TYPE(affinity_str);
    std::string device_str = devicePtr;
    DEVICE_TYPE device_cur = str2DEVICE_TYPE(device_str);

    long modelAddr = (long)model_create(modelPathPtr, affinity_cur, device_cur);     
    return modelAddr;
}

extern "C" JNIEXPORT void JNICALL Java_BoltModel_model_1ready
  (JNIEnv *env, jobject, jlong modelAddr, jint num_input, jobjectArray input_names, jintArray n, jintArray c, jintArray h, jintArray w, jobjectArray dt_input, jobjectArray df_input) {
    IHandle ih = (IHandle)modelAddr;    

    jint *curArray_n = env->GetIntArrayElements(n, 0);
    int* datas_n = (int*)malloc(num_input * sizeof(int));
    jint *curArray_c = env->GetIntArrayElements(c, 0);
    int* datas_c = (int*)malloc(num_input * sizeof(int));
    jint *curArray_h = env->GetIntArrayElements(h, 0);
    int* datas_h = (int*)malloc(num_input * sizeof(int));
    jint *curArray_w = env->GetIntArrayElements(w, 0);
    int* datas_w = (int*)malloc(num_input * sizeof(int));
    for (int i = 0; i < num_input; i++) {
	datas_n[i] = curArray_n[i];
	datas_c[i] = curArray_c[i];
	datas_h[i] = curArray_h[i];
	datas_w[i] = curArray_w[i];
    }

    char** input_names_ptr = (char**)malloc(sizeof(char*) * num_input);
    std::vector<std::string> name_strs;   
    for (int i=0; i < num_input; i++) {
	jstring cur_str = (jstring)(env->GetObjectArrayElement(input_names, i));
	const char* cur_str_ptr = env->GetStringUTFChars(cur_str, 0);
	std::string tmp_str = cur_str_ptr;
	name_strs.push_back(tmp_str);
	input_names_ptr[i] = (char*)name_strs[i].c_str();
	
	env->ReleaseStringUTFChars(cur_str, cur_str_ptr);
	env->DeleteLocalRef(cur_str); 	
    }    

    for (int i=0; i<num_input; i++) {
	input_names_ptr[i] = (char*)(name_strs[i].c_str());
    }

    DATA_TYPE* dt_inputs_ptr = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * num_input);
    DATA_FORMAT* df_inputs_ptr = (DATA_FORMAT*)malloc(sizeof(DATA_FORMAT) * num_input);
    int dt_input_num = env->GetArrayLength(dt_input);
    int df_input_num = env->GetArrayLength(df_input);

    if (dt_input_num != df_input_num) {
	    std::cerr << "[ERROR]: num of input_datatype not equal to num of input_dataformat!" << std::endl;
        exit(1);
    }

    for (int i=0; i<dt_input_num; i++) {
	jstring tmp_str_dt = (jstring)(env->GetObjectArrayElement(dt_input, i));
	const char* tmp_str_dt_ptr = env->GetStringUTFChars(tmp_str_dt, 0);
	std::string cur_tmp_str_dt = tmp_str_dt_ptr;
	dt_inputs_ptr[i] = str2DATA_TYPE(cur_tmp_str_dt);

        jstring tmp_str_df = (jstring)(env->GetObjectArrayElement(df_input, i));
	const char* tmp_str_df_ptr = env->GetStringUTFChars(tmp_str_df, 0);
	std::string cur_tmp_str_df = tmp_str_df_ptr;
	df_inputs_ptr[i] = str2DATA_FORMAT(cur_tmp_str_df);	
    }

    model_ready(ih, num_input, datas_n, datas_c, datas_h, datas_w, input_names_ptr, dt_inputs_ptr, df_inputs_ptr);

    env->ReleaseIntArrayElements(n, curArray_n, 0);
    free(datas_n);
    env->ReleaseIntArrayElements(c, curArray_c, 0);
    free(datas_c);
    env->ReleaseIntArrayElements(h, curArray_h, 0);
    free(datas_h);
    env->ReleaseIntArrayElements(w, curArray_w, 0);
    free(datas_w);
    free(input_names_ptr);
}

extern "C" JNIEXPORT void JNICALL Java_BoltModel_model_1resize_1input
  (JNIEnv *env, jobject, jlong modelAddr, jint num_input, jobjectArray input_names, jintArray n, jintArray c, jintArray h, jintArray w, jobjectArray dt_input, jobjectArray df_input) {
    IHandle ih = (IHandle)modelAddr;

    jint *curArray_n = env->GetIntArrayElements(n, 0);
    int* datas_n = (int*)malloc(num_input * sizeof(int));
    jint *curArray_c = env->GetIntArrayElements(c, 0);
    int* datas_c = (int*)malloc(num_input * sizeof(int));
    jint *curArray_h = env->GetIntArrayElements(h, 0);
    int* datas_h = (int*)malloc(num_input * sizeof(int));
    jint *curArray_w = env->GetIntArrayElements(w, 0);
    int* datas_w = (int*)malloc(num_input * sizeof(int));
    for (int i = 0; i < num_input; i++) {
        datas_n[i] = curArray_n[i];
        datas_c[i] = curArray_c[i];
        datas_h[i] = curArray_h[i];
        datas_w[i] = curArray_w[i];
    }

    char** input_names_ptr = (char**)malloc(sizeof(char*) * num_input);
    std::vector<std::string> name_strs;
    for (int i=0; i < num_input; i++) {
        jstring cur_str = (jstring)(env->GetObjectArrayElement(input_names, i));
        const char* cur_str_ptr = env->GetStringUTFChars(cur_str, 0);
        std::string tmp_str = cur_str_ptr;
        name_strs.push_back(tmp_str);
        input_names_ptr[i] = (char*)name_strs[i].c_str();

        env->ReleaseStringUTFChars(cur_str, cur_str_ptr);
        env->DeleteLocalRef(cur_str);
    }

    for (int i=0; i<num_input; i++) {
        input_names_ptr[i] = (char*)(name_strs[i].c_str());
    }

    DATA_TYPE* dt_inputs_ptr = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * num_input);
    DATA_FORMAT* df_inputs_ptr = (DATA_FORMAT*)malloc(sizeof(DATA_FORMAT) * num_input);
    int dt_input_num = env->GetArrayLength(dt_input);
    int df_input_num = env->GetArrayLength(df_input);

    if (dt_input_num != df_input_num) {
        std::cerr << "[ERROR]: num of input_datatype not equal to num of input_dataformat!" << std::endl;
        exit(1);
    }

    for (int i=0; i<dt_input_num; i++) {
        jstring tmp_str_dt = (jstring)(env->GetObjectArrayElement(dt_input, i));
        const char* tmp_str_dt_ptr = env->GetStringUTFChars(tmp_str_dt, 0);
        std::string cur_tmp_str_dt = tmp_str_dt_ptr;
        dt_inputs_ptr[i] = str2DATA_TYPE(cur_tmp_str_dt);

        jstring tmp_str_df = (jstring)(env->GetObjectArrayElement(df_input, i));
        const char* tmp_str_df_ptr = env->GetStringUTFChars(tmp_str_df, 0);
        std::string cur_tmp_str_df = tmp_str_df_ptr;
        df_inputs_ptr[i] = str2DATA_FORMAT(cur_tmp_str_df);
    }

    model_resize_input(ih, num_input, datas_n, datas_c, datas_h, datas_w, input_names_ptr, dt_inputs_ptr, df_inputs_ptr);

    env->ReleaseIntArrayElements(n, curArray_n, 0);
    free(datas_n);
    env->ReleaseIntArrayElements(c, curArray_c, 0);
    free(datas_c);
    env->ReleaseIntArrayElements(h, curArray_h, 0);
    free(datas_h);
    env->ReleaseIntArrayElements(w, curArray_w, 0);
    free(datas_w);
    free(input_names_ptr);
}

extern "C" JNIEXPORT jlong JNICALL Java_BoltModel_IResult_1malloc_1all
  (JNIEnv *, jobject, jlong modelAddr) {
    IHandle ih = (IHandle)modelAddr;
    IResult ir = IResult_malloc_all(ih);
    return (long)ir;
}

extern "C" JNIEXPORT jlong JNICALL Java_BoltModel_IResult_1malloc_1part
  (JNIEnv *env, jobject, jlong modelAddr, jint num_outputs, jobjectArray outputNames) {
    IHandle ih = (IHandle)modelAddr;
    char** output_names_ptr = (char**)malloc(sizeof(char*) * num_outputs);
    std::vector<std::string> name_strs;
    for (int i=0; i < num_outputs; i++) {
        jstring cur_str = (jstring)(env->GetObjectArrayElement(outputNames, i));
        const char* cur_str_ptr = env->GetStringUTFChars(cur_str, 0);
        std::string tmp_str = cur_str_ptr;
        name_strs.push_back(tmp_str);
        output_names_ptr[i] = (char*)name_strs[i].c_str();

        env->ReleaseStringUTFChars(cur_str, cur_str_ptr);
        env->DeleteLocalRef(cur_str);
    }

    for (int i=0; i<num_outputs; i++) {
        output_names_ptr[i] = (char*)(name_strs[i].c_str());
    }
    IResult ir = IResult_malloc_part(ih, num_outputs, output_names_ptr);

    free(output_names_ptr);
    return (long)ir;
}

extern "C" JNIEXPORT void JNICALL Java_BoltModel_model_1run
  (JNIEnv *env, jobject, jlong modelAddr, jlong IResultAddr, jint num_input, jobjectArray input_names, jobjectArray inputData) {
    IHandle ih = (IHandle)modelAddr;
    IResult ir = (IResult)IResultAddr;
    
    IHandleInfo* ihInfo = (IHandleInfo*)ih;
    CNN* cnn = (CNN*)ihInfo->cnn;

    char** input_names_ptr = (char**)malloc(sizeof(char*) * num_input);
    std::vector<std::string> name_strs;
    for (int i=0; i < num_input; i++) {
        jstring cur_str = (jstring)(env->GetObjectArrayElement(input_names, i));
        const char* cur_str_ptr = env->GetStringUTFChars(cur_str, 0);
        std::string tmp_str = cur_str_ptr;
        name_strs.push_back(tmp_str);
        input_names_ptr[i] = (char*)name_strs[i].c_str();
        env->ReleaseStringUTFChars(cur_str, cur_str_ptr);
        env->DeleteLocalRef(cur_str);
    }

    for (int i=0; i<num_input; i++) {
	input_names_ptr[i] = (char*)(name_strs[i].c_str());
    }

    void** mem_ptr = (void**)malloc(sizeof(void*) * num_input);
    jint rows = env->GetArrayLength(inputData);
    HashMap<std::string, std::shared_ptr<Tensor>> inMap = cnn->get_inputs();
    for (int i=0; i<rows; i++) {
	jfloatArray curArray = static_cast<jfloatArray>(env->GetObjectArrayElement(inputData, i));
	jfloat* datas = env->GetFloatArrayElements(curArray, JNI_FALSE);
	std::string curTensorName = name_strs[i];
	std::shared_ptr<Tensor> cur_input_tensor = inMap[curTensorName];
	jint clos = env->GetArrayLength(curArray);
	
        TensorDesc tensorDesc = cur_input_tensor->get_desc();
        mem_ptr[i] = cur_input_tensor->get_val();
        dataTypeConverterFromFloat(datas, mem_ptr[i], tensorDesc.dt, clos);
    }

    model_run(ih, ir, num_input, input_names_ptr, mem_ptr);
    free(input_names_ptr);
    free(mem_ptr);
}

int calculateLength(int *array, int num) {
    int length = 0;
    for (int j = 0; j < num; j++) {
        if (array[j] == 0)
            break;
        else {
            if (length == 0)
                length = array[j];
            else
                length *= array[j];
        }
    }
    return length;
}

extern "C" JNIEXPORT jobject JNICALL Java_BoltModel_getOutput
  (JNIEnv *env, jobject, jlong IResultAddr) {
    jclass stucls = env->FindClass("BoltResult");
   
    jmethodID constrocMID = env->GetMethodID(stucls, "<init>", "([[F[[I[Ljava/lang/String;[Ljava/lang/String;)V");
    
    IResultInner* ir_inner = (IResultInner*)IResultAddr;
    DataDesc* outputArrPtr = (*ir_inner).outputArr;
    int num_outputs = (*ir_inner).num_outputs;

    jobjectArray output_values;
    jclass floatArrCls = env->FindClass("[F");
    output_values = env->NewObjectArray(num_outputs, floatArrCls, nullptr);
    jobjectArray output_dimension;
    jclass intArrCls = env->FindClass("[I");
    output_dimension = env->NewObjectArray(num_outputs, intArrCls, nullptr);

    jobjectArray output_names_arr;
    output_names_arr = (jobjectArray)env->NewObjectArray(num_outputs, env->FindClass("java/lang/String"), env->NewStringUTF(""));

    jobjectArray df_arr;
    df_arr = (jobjectArray)env->NewObjectArray(num_outputs, env->FindClass("java/lang/String"), env->NewStringUTF(""));

    for (int i=0; i<num_outputs; i++) {
	std::string cur_output_name = outputArrPtr[i].name;
	env->SetObjectArrayElement(output_names_arr, i, env->NewStringUTF(cur_output_name.c_str()));
	
	DataType cur_data_type = outputArrPtr[i].dt;

	DataFormat cur_data_format = outputArrPtr[i].df;
	std::string cur_data_format_str = DataFormat2str(cur_data_format);
	env->SetObjectArrayElement(df_arr, i, env->NewStringUTF(cur_data_format_str.c_str()));

	void* cur_dataPtr = outputArrPtr[i].dataPtr;
	int tensorNumber = calculateLength((int*)outputArrPtr[i].dims, 4);
	jfloat tmp_output_values[tensorNumber];
	jfloatArray floatArr = env->NewFloatArray(tensorNumber);

        jint tmp_output_dimensions[4];
	jintArray intArr = env->NewIntArray(4);

	for (int j = 0; j < 4; j++) {
	    tmp_output_dimensions[j] = (int)(outputArrPtr[i].dims[j]);
	}

        dataTypeConverterToFloat(cur_dataPtr, cur_data_type, tmp_output_values, tensorNumber);
	env->SetFloatArrayRegion(floatArr, 0, tensorNumber, tmp_output_values);
	env->SetObjectArrayElement(output_values, i, floatArr);
	env->DeleteLocalRef(floatArr);

        env->SetIntArrayRegion(intArr, 0, 4, tmp_output_dimensions);
        env->SetObjectArrayElement(output_dimension, i, intArr);
        env->DeleteLocalRef(intArr);	
    }   

    jobject bolt_result_obj = env->NewObject(stucls, constrocMID, output_values, output_dimension, output_names_arr, df_arr);
    return bolt_result_obj;
}

extern "C" JNIEXPORT void JNICALL Java_BoltModel_IResult_1free
  (JNIEnv *, jobject, jlong IResultAddr) {
    IResult ir = (IResult)IResultAddr;
    IResult_free(ir);
}

extern "C" JNIEXPORT void JNICALL Java_BoltModel_destroyModel
  (JNIEnv *, jobject, jlong modelAddr) {
    IHandle ih = (IHandle)modelAddr;
    model_destroy(ih);
}
#endif
