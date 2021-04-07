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
#include <string>
#include "libbolt/headers/kit_flags.h"
#include "libbolt/headers/flow.h"
#include <android/log.h>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "========= Info =========   ", __VA_ARGS__)

DataType inferencePrecision = DT_F32;
Flow flowExample;
const int width = 32;
const int height = 32;

EE pixelProcess(std::map<std::string, std::shared_ptr<Tensor>> &inputs,
    std::shared_ptr<Tensor> &tmp,
    std::map<std::string, std::shared_ptr<Tensor>> &outputs,
    std::vector<std::string> parameter = std::vector<std::string>())
{
    // Already in BGR
    uint8_t *myBuffer = (uint8_t *)((CpuMemory *)inputs["input"]->get_memory())->get_ptr();

    F32 *oneArr = (F32 *)((CpuMemory *)outputs["input.1"]->get_memory())->get_ptr();

    for (int i = 0; i < width * height * 3; i++) {
        oneArr[i] = myBuffer[i];
    }
    return SUCCESS;
}

EE postProcess(std::map<std::string, std::shared_ptr<Tensor>> &inputs,
    std::shared_ptr<Tensor> &tmp,
    std::map<std::string, std::shared_ptr<Tensor>> &outputs,
    std::vector<std::string> parameter = std::vector<std::string>())
{
    std::string flowInferenceNodeOutputName = "output";
    std::string boltModelOutputName = "1811";

    uint8_t *flowInferenceNodeOutput =
        (uint8_t *)((CpuMemory *)outputs[flowInferenceNodeOutputName]->get_memory())->get_ptr();
    F32 *result = (F32 *)((CpuMemory *)inputs[boltModelOutputName]->get_memory())->get_ptr();

    F32 *rArr = (F32 *)malloc(sizeof(int *) * width * 2 * height * 2);
    F32 *gArr = (F32 *)malloc(sizeof(int *) * width * 2 * height * 2);
    F32 *bArr = (F32 *)malloc(sizeof(int *) * width * 2 * height * 2);
    for (int i = 0; i < (height * 2) * (width * 2) * 3; i++) {
        if (result[i] <= 1) {
            int a = 0;
            result[i] = a;
        } else if (result[i] > 255) {
            int b = 255;
            result[i] = b;
        }

        if (i < (width * 2) * (height * 2)) {
            bArr[i] = result[i];
        } else if (i < (width * 2) * (height * 2) * 2) {
            rArr[i - (width * 2) * (height * 2)] = result[i];
        } else {
            gArr[i - 2 * (width * 2) * (height * 2)] = result[i];
        }
    }

    for (int i = 0; i < (width * 2) * (height * 2); i++) {
        int r = rArr[i];
        int g = gArr[i];
        int b = bArr[i];

        flowInferenceNodeOutput[i * 3] = (uint8_t)r;
        flowInferenceNodeOutput[i * 3 + 1] = (uint8_t)g;
        flowInferenceNodeOutput[i * 3 + 2] = (uint8_t)b;
    }

    free(rArr);
    free(gArr);
    free(bArr);
    return SUCCESS;
}

extern "C" void Java_com_example_cameraenlarge_MainActivity_initFlow(
    JNIEnv *env, jobject, jstring path)
{
    flowRegisterFunction("pixelProcess", pixelProcess);
    flowRegisterFunction("postProcess", postProcess);

    const char *cur_str_path = env->GetStringUTFChars(path, nullptr);
    std::string imageClassificationGraphPath = cur_str_path;
    std::vector<std::string> graphPath = {imageClassificationGraphPath};
    int threads = 1;

    flowExample.init(graphPath, inferencePrecision, AFFINITY_CPU_HIGH_PERFORMANCE, threads, false);
}

std::map<std::string, std::shared_ptr<Tensor>> inputOutput(const unsigned char *myBuffer)
{
    std::map<std::string, std::shared_ptr<Tensor>> tensors;
    TensorDesc inputDesc = tensor4df(DT_U8, DF_NCHW, 1, 3, height, width);

    tensors["input"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["input"]->resize(inputDesc);
    tensors["input"]->alloc();
    void *ptr = (void *)((CpuMemory *)tensors["input"]->get_memory())->get_ptr();
    memcpy(ptr, myBuffer, tensorNumBytes(inputDesc));

    tensors["output"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["output"]->resize(tensor2df(DT_I32, DF_NCHW, 1, (width * 2) * (height * 2) * 3));
    tensors["output"]->alloc();
    return tensors;
}

extern "C" jbyteArray Java_com_example_cameraenlarge_MainActivity_runFlow(
    JNIEnv *env, jobject, jbyteArray bgrData, jstring path)
{
    int len = env->GetArrayLength(bgrData);
    unsigned char *theValue = new unsigned char[len];
    env->GetByteArrayRegion(bgrData, 0, len, reinterpret_cast<jbyte *>(theValue));

    int num = 1;
    const char *cur_str_path = env->GetStringUTFChars(path, nullptr);
    std::string imageClassificationGraphPath = cur_str_path;

    for (int i = 0; i < num; i++) {
        std::map<std::string, std::shared_ptr<Tensor>> data = inputOutput(theValue);
        Task task(imageClassificationGraphPath, data);
        flowExample.enqueue(task);
    }
    std::vector<Task> results;

    UNI_PROFILE(results = flowExample.dequeue(true), std::string("image_classification"),
        std::string("image_classification"));

    uint8_t *result = (uint8_t *)((CpuMemory *)results[0].data["output"]->get_memory())->get_ptr();

    uint8_t *endResult = (uint8_t *)malloc(sizeof(uint8_t *) * (width * 2) * (height * 2) * 4);
    for (int i = 0; i < (width * 2) * (height * 2) * 4; i++) {  //RGBA  add alpha
        if (i % 4 != 3) {
            endResult[i] = result[i - (i / 4) - 1];
        } else {
            int alpha = 255;
            endResult[i] = (unsigned char)alpha;
        }
    }
    jbyteArray intArr = env->NewByteArray((width * 2) * (height * 2) * 4);

    env->SetByteArrayRegion(
        intArr, 0, (width * 2) * (height * 2) * 4, reinterpret_cast<jbyte *>(endResult));
    free(endResult);
    return intArr;
}
