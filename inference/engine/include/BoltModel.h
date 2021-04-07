// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _Included_BoltModel
#define _Included_BoltModel
#ifdef __cplusplus
extern "C" {
#endif

#include "jni_header.h"

/*
 * Class:     BoltModel
 * Method:    createModel
 * Signature: (Ljava/lang/String;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL BOLT_JNI_PREFIX(BoltModel_createModel)(JNIEnv *, jobject, jstring, jstring);

/*
 * Class:     BoltModel
 * Method:    prepareModel
 * Signature: (JI[Ljava/lang/String;[I[I[I[I[Ljava/lang/String;[Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL BOLT_JNI_PREFIX(BoltModel_prepareModel)(JNIEnv *,
    jobject,
    jlong,
    jint,
    jobjectArray,
    jintArray,
    jintArray,
    jintArray,
    jintArray,
    jobjectArray,
    jobjectArray);

/*
 * Class:     BoltModel
 * Method:    cloneModel
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL BOLT_JNI_PREFIX(BoltModel_cloneModel)(JNIEnv *, jobject, jlong);

/*
 * Class:     BoltModel
 * Method:    resizeModelInput
 * Signature: (JI[Ljava/lang/String;[I[I[I[I[Ljava/lang/String;[Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL BOLT_JNI_PREFIX(BoltModel_resizeModelInput)(JNIEnv *,
    jobject,
    jlong,
    jint,
    jobjectArray,
    jintArray,
    jintArray,
    jintArray,
    jintArray,
    jobjectArray,
    jobjectArray);

/*
 * Class:     BoltModel
 * Method:    allocAllResultHandle
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL BOLT_JNI_PREFIX(BoltModel_allocAllResultHandle)(JNIEnv *, jobject, jlong);

/*
 * Class:     BoltModel
 * Method:    allocSpecificResultHandle
 * Signature: (JI[Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL BOLT_JNI_PREFIX(BoltModel_allocSpecificResultHandle)(
    JNIEnv *, jobject, jlong, jint, jobjectArray);

/*
 * Class:     BoltModel
 * Method:    cloneResult
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL BOLT_JNI_PREFIX(BoltModel_cloneResultHandle)(JNIEnv *, jobject, jlong);

/*
 * Class:     BoltModel
 * Method:    setRuntimeDeviceJNI
 * Signature: (ILjava/lang/String;)V
 */
JNIEXPORT void JNICALL BOLT_JNI_PREFIX(BoltModel_setRuntimeDeviceJNI)(
    JNIEnv *, jobject, jlong, jint, jstring);

/*
 * Class:     BoltModel
 * Method:    setRuntimeDeviceDynamicJNI
 * Signature: (V)V
 */
JNIEXPORT void JNICALL BOLT_JNI_PREFIX(BoltModel_setRuntimeDeviceDynamicJNI)(
    JNIEnv *, jobject, jlong);

/*
 * Class:     BoltModel
 * Method:    runModel
 * Signature: (JJI[Ljava/lang/String;[[F)V
 */
JNIEXPORT void JNICALL BOLT_JNI_PREFIX(BoltModel_runModel)(
    JNIEnv *, jobject, jlong, jlong, jint, jobjectArray, jobjectArray);

/*
 * Class:     BoltModel
 * Method:    getOutput
 * Signature: (J)LBoltResult;
 */
JNIEXPORT jobject JNICALL BOLT_JNI_PREFIX(BoltModel_getOutput)(JNIEnv *, jobject, jlong, jstring);

/*
 * Class:     BoltModel
 * Method:    freeResultHandle
 * Signature: (J)V
 */
JNIEXPORT void JNICALL BOLT_JNI_PREFIX(BoltModel_freeResultHandle)(JNIEnv *, jobject, jlong);

/*
 * Class:     BoltModel
 * Method:    destroyModel
 * Signature: (J)V
 */
JNIEXPORT void JNICALL BOLT_JNI_PREFIX(BoltModel_destroyModel)(JNIEnv *, jobject, jlong);

#ifdef __cplusplus
}
#endif
#endif
