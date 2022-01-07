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

DataType inferencePrecision = DT_F32;
Flow flowExample;
const int topK = 5;
const int width = 224;
const int height = 224;
double useTime;

EE pixelProcess(std::map<std::string, std::shared_ptr<Tensor>> &inputs,
    std::shared_ptr<Tensor> &tmp,
    std::map<std::string, std::shared_ptr<Tensor>> &outputs,
    std::vector<std::string> parameter = std::vector<std::string>())
{
    // Already in BGR
    unsigned char *myBuffer =
        (unsigned char *)((CpuMemory *)inputs["input:1"]->get_memory())->get_ptr();

    F32 *oneArr = (F32 *)((CpuMemory *)outputs["input:0"]->get_memory())->get_ptr();
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
    std::string boltModelOutputName = "MobileNetV2/Predictions/Softmax:0";

    int *flowInferenceNodeOutput =
        (int *)((CpuMemory *)outputs[flowInferenceNodeOutputName]->get_memory())->get_ptr();
    F32 *score1000 = (F32 *)((CpuMemory *)inputs[boltModelOutputName]->get_memory())->get_ptr();

    for (int i = 0; i < topK; i++) {
        int max_index = 0;
        for (int j = 1; j < 1000; j++) {
            if (score1000[j] > score1000[max_index]) {
                max_index = j;
            }
        }
        flowInferenceNodeOutput[i] = max_index;
        score1000[max_index] = -65504;
    }
    return SUCCESS;
}

extern "C" void Java_com_example_imageclassificationapp_MainActivity_initFlow(
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
    TensorDesc inputDesc = tensor4df(DT_U8, DF_NCHW, 1, 224, 224, 3);

    tensors["input:1"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["input:1"]->resize(inputDesc);
    tensors["input:1"]->alloc();
    void *ptr = (void *)((CpuMemory *)tensors["input:1"]->get_memory())->get_ptr();
    memcpy(ptr, myBuffer, tensorNumBytes(inputDesc));

    tensors["output"] = std::shared_ptr<Tensor>(new Tensor());
    tensors["output"]->resize(tensor2df(DT_I32, DF_NCHW, 1, topK));
    tensors["output"]->alloc();
    return tensors;
}

extern "C" jintArray Java_com_example_imageclassificationapp_MainActivity_runFlow(
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

    double start = ut_time_ms();
    UNI_PROFILE(results = flowExample.dequeue(true), std::string("image_classification"),
        std::string("image_classification"));
    double end = ut_time_ms();
    useTime = end - start;

    int *top5 = (int *)((CpuMemory *)results[0].data["output"]->get_memory())->get_ptr();
    jintArray intArr = env->NewIntArray(topK);
    jint buf[topK];
    for (int i = 0; i < topK; ++i) {
        buf[i] = top5[i];
    }
    env->SetIntArrayRegion(intArr, 0, topK, buf);
    return intArr;
}

extern "C" jdouble Java_com_example_imageclassificationapp_MainActivity_useTimeFromJNI(
    JNIEnv *env, jobject)
{
    return useTime;
}
