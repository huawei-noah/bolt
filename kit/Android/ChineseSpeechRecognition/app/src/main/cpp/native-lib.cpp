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

#include "libbolt/headers/kit_flags.h"
#include "flow_asr.h"

extern "C" JNIEXPORT void JNICALL Java_com_huawei_noah_MainActivity_initFlow(JNIEnv *env,
    jobject thiz,
    jstring encoder_path,
    jstring predic_path,
    jstring joint_path,
    jstring pinyin_path,
    jstring label_path)
{
    encoderGraphPath = env->GetStringUTFChars(encoder_path, nullptr);
    predictionGraphPath = env->GetStringUTFChars(predic_path, nullptr);
    jointGraphPath = env->GetStringUTFChars(joint_path, nullptr);
    pinyin2hanziGraphPath = env->GetStringUTFChars(pinyin_path, nullptr);
    labelFilePath = env->GetStringUTFChars(label_path, nullptr);

    initASRFlow();
}

extern "C" JNIEXPORT jstring JNICALL Java_com_huawei_noah_MainActivity_runFlow(
    JNIEnv *env, jobject thiz, jstring wav_file_path)
{
    std::string wavFilePath = env->GetStringUTFChars(wav_file_path, nullptr);
    std::string hanzi = runASRFlow(wavFilePath);
    return env->NewStringUTF(hanzi.c_str());
}
