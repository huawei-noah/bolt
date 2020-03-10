// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <jni.h>
/* Header for class BoltModel */

#ifndef _Included_BoltModel
#define _Included_BoltModel
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     BoltModel
 * Method:    model_create
 * Signature: (Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_BoltModel_model_1create
  (JNIEnv *, jobject, jstring, jstring, jstring);

/*
 * Class:     BoltModel
 * Method:    model_ready
 * Signature: (JI[Ljava/lang/String;[I[I[I[I[Ljava/lang/String;[Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_BoltModel_model_1ready
  (JNIEnv *, jobject, jlong, jint, jobjectArray, jintArray, jintArray, jintArray, jintArray, jobjectArray, jobjectArray);

/*
 * Class:     BoltModel
 * Method:    model_ready
 * Signature: (JI[Ljava/lang/String;[I[I[I[I[Ljava/lang/String;[Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_BoltModel_model_1resize_1input
  (JNIEnv *, jobject, jlong, jint, jobjectArray, jintArray, jintArray, jintArray, jintArray, jobjectArray, jobjectArray);

/*
 * Class:     BoltModel
 * Method:    IResult_malloc_all
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_BoltModel_IResult_1malloc_1all
  (JNIEnv *, jobject, jlong);

/*
 * Class:     BoltModel
 * Method:    IResult_malloc_part
 * Signature: (JI[Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_BoltModel_IResult_1malloc_1part
  (JNIEnv *, jobject, jlong, jint, jobjectArray);

/*
 * Class:     BoltModel
 * Method:    model_run
 * Signature: (JJI[Ljava/lang/String;[[F)V
 */
JNIEXPORT void JNICALL Java_BoltModel_model_1run
  (JNIEnv *, jobject, jlong, jlong, jint, jobjectArray, jobjectArray);

/*
 * Class:     BoltModel
 * Method:    getOutput
 * Signature: (J)LBoltResult;
 */
JNIEXPORT jobject JNICALL Java_BoltModel_getOutput
  (JNIEnv *, jobject, jlong);

/*
 * Class:     BoltModel
 * Method:    IResult_free
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_BoltModel_IResult_1free
  (JNIEnv *, jobject, jlong);

/*
 * Class:     BoltModel
 * Method:    destroyModel
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_BoltModel_destroyModel
  (JNIEnv *, jobject, jlong);

#ifdef __cplusplus
}
#endif
#endif
