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
#include "libbolt/headers/ultra_face.h"
#include <getopt.h>
#include "libbolt/headers/opencv2/core/core.hpp"
#include "libbolt/headers/opencv2/highgui/highgui.hpp"
#include "libbolt/headers/opencv2/imgproc/imgproc.hpp"
#include "libbolt/headers/opencv2/opencv.hpp"
#include <android/log.h>

#include <android/bitmap.h>

using namespace cv;

char *modelPath = (char *)"";
std::string inputData = "";
char *affinityPolicyName = (char *)"CPU_AFFINITY_HIGH_PERFORMANCE";
char *algorithmMapPath = (char *)"";
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "========= Info =========   ", __VA_ARGS__)

#define ASSERT(status, ret) \
    if (!(status)) {        \
        return ret;         \
    }
#define ASSERT_FALSE(status) ASSERT(status, false)

bool BitmapToMatrix(JNIEnv *env, jobject obj_bitmap, cv::Mat &matrix)
{
    void *bitmapPixels;            // Save picture pixel data
    AndroidBitmapInfo bitmapInfo;  // Save picture parameters

    ASSERT_FALSE(AndroidBitmap_getInfo(env, obj_bitmap, &bitmapInfo) >= 0);  // Get picture parameters
    ASSERT_FALSE(bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
        bitmapInfo.format ==
            ANDROID_BITMAP_FORMAT_RGB_565);  // Only ARGB? 8888 and RGB? 565 are supported
    ASSERT_FALSE(AndroidBitmap_lockPixels(env, obj_bitmap, &bitmapPixels) >=
        0);  // Get picture pixels (lock memory block)
    ASSERT_FALSE(bitmapPixels);

    if (bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        cv::Mat tmp(
            bitmapInfo.height, bitmapInfo.width, CV_8UC4, bitmapPixels);  // Establish temporary mat
        tmp.copyTo(matrix);                                               // Copy to target matrix
    } else {
        cv::Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC2, bitmapPixels);
        cv::cvtColor(tmp, matrix, cv::COLOR_BGR5652RGB);
    }

    //convert RGB to BGR
    cv::cvtColor(matrix, matrix, cv::COLOR_RGB2BGR);

    AndroidBitmap_unlockPixels(env, obj_bitmap);  // Unlock
    return true;
}

std::map<std::string, std::shared_ptr<Tensor>> get_output(
    std::shared_ptr<CNN> pipeline, std::string affinity)
{
    std::map<std::string, std::shared_ptr<Tensor>> outMap = pipeline->get_output();
    if (affinity == "GPU") {
#ifdef _USE_MALI
        for (auto iter : outMap) {
            Tensor result = *(iter.second);
            auto mem = (OclMemory *)result.get_memory();
            mem->get_mapped_ptr();
        }
#else
        UNI_WARNING_LOG("this binary not support GPU, please recompile project with GPU "
                        "compile options\n");
#endif
    }
    return outMap;
}

extern "C" JNIEXPORT void JNICALL Java_com_huawei_noah_BoltResult_initBolt(JNIEnv *env, jobject thiz)
{
    // TODO: implement initBolt()
    prior_boxes_generator(320, 240, 0.7, 0.3);
}

extern "C" JNIEXPORT jstring JNICALL Java_com_huawei_noah_BoltResult_runBolt(
    JNIEnv *env, jobject thiz, jint num_input, jstring model_path, jstring photo_path)
{
    // TODO: implement runBolt()
    std::string modelFilePath = env->GetStringUTFChars(model_path, nullptr);
    modelPath = (char *)modelFilePath.c_str();

    inputData = env->GetStringUTFChars(photo_path, nullptr);

    int last_gang_index = inputData.find_last_of('/');
    int last_dot_index = inputData.find_last_of('.');
    std::string prefix_str = inputData.substr(0, last_gang_index + 1);
    std::string data_name =
        inputData.substr(last_gang_index + 1, last_dot_index - last_gang_index - 1);
    std::string suffix_str = inputData.substr(last_dot_index, inputData.length());
    std::string output_data_path = prefix_str + "bolt_" + data_name + suffix_str;

    cv::Mat img;
    img = imread(inputData);
    image_h = img.rows;                // global variable
    image_w = img.cols;                // global variable
    int img_channel = img.channels();  // local variable
    cv::Mat img_float;
    cv::Mat img_resize;
    std::vector<float> vec_original;
    std::shared_ptr<U8> input_ptr(new U8[image_h * image_w * img_channel * sizeof(float)]);
    float *vec_normalize = (float *)(input_ptr.get());
    auto pipeline = createPipeline(affinityPolicyName, modelPath, algorithmMapPath);
    std::map<std::string, TensorDesc> inputDescMap = pipeline->get_input_desc();
    auto item = inputDescMap.begin();
    std::map<std::string, std::shared_ptr<U8>> model_tensors_input;
    std::map<std::string, std::shared_ptr<Tensor>> outMap;

    while (1) {
        img.convertTo(img_float, CV_32F);
        cv::resize(img_float, img_resize, cv::Size(320, 240));  // magic number
        vec_original.assign((float *)img_resize.datastart, (float *)img_resize.dataend);
        int iter_index = 0;
        for (int i = img.channels() - 1; i >= 0; i--) {
            for (unsigned int j = 0; j < vec_original.size() / img.channels(); j++) {
                vec_normalize[iter_index] = (vec_original[j * img.channels() + i] - 127.0) / 128.0;
                iter_index++;
            }
        }
        model_tensors_input[item->first] = input_ptr;
        pipeline->set_input_by_assign(model_tensors_input);
        pipeline->run();
        outMap = get_output(pipeline, affinityPolicyName);
        std::vector<FaceInfo> bbox_collection;
        Tensor box_tensor = *(outMap["boxes"].get());
        Tensor score_tensor = *(outMap["scores"].get());
        bounding_boxes_generator(bbox_collection, box_tensor, score_tensor);
        std::vector<FaceInfo> bolt_final_result;
        nms(bbox_collection, bolt_final_result, hard_nms);
        for (unsigned int i = 0; i < bolt_final_result.size(); i++) {
            auto face = bolt_final_result[i];
            cv::Point pt1(face.x1, face.y1);
            cv::Point pt2(face.x2, face.y2);
            cv::rectangle(img, pt1, pt2, cv::Scalar(0, 255, 0), 2);
        }
        cv::imwrite(output_data_path, img);
        break;
    }
    LOGI("result saved at=%s", output_data_path.c_str());

    return env->NewStringUTF(output_data_path.c_str());
}

extern "C" JNIEXPORT jstring JNICALL Java_com_huawei_noah_BoltResult_runBolt2(
    JNIEnv *env, jobject thiz, jint num_input, jobject bitmap, jstring model_path)
{
    // TODO: implement runBolt2()

    std::string modelFilePath = env->GetStringUTFChars(model_path, nullptr);
    modelPath = (char *)modelFilePath.c_str();

    std::string output_data_path = "/data/user/0/com.huawei.noah/cache/result_bolt.jpg";

    cv::Mat img;
    BitmapToMatrix(env, bitmap, img);
    image_h = img.rows;                // global variable
    image_w = img.cols;                // global variable
    int img_channel = img.channels();  // local variable
    cv::Mat img_float;
    cv::Mat img_resize;
    std::vector<float> vec_original;
    std::shared_ptr<U8> input_ptr(new U8[image_h * image_w * img_channel * sizeof(float)]);
    float *vec_normalize = (float *)(input_ptr.get());
    auto pipeline = createPipeline(affinityPolicyName, modelPath, algorithmMapPath);
    std::map<std::string, TensorDesc> inputDescMap = pipeline->get_input_desc();
    auto item = inputDescMap.begin();
    std::map<std::string, std::shared_ptr<U8>> model_tensors_input;
    std::map<std::string, std::shared_ptr<Tensor>> outMap;

    while (1) {
        img.convertTo(img_float, CV_32F);
        cv::resize(img_float, img_resize, cv::Size(320, 240));  // magic number
        vec_original.assign((float *)img_resize.datastart, (float *)img_resize.dataend);
        int iter_index = 0;
        for (int i = img.channels() - 1; i >= 0; i--) {
            for (unsigned int j = 0; j < vec_original.size() / img.channels(); j++) {
                vec_normalize[iter_index] = (vec_original[j * img.channels() + i] - 127.0) / 128.0;
                iter_index++;
            }
        }
        model_tensors_input[item->first] = input_ptr;
        pipeline->set_input_by_assign(model_tensors_input);
        pipeline->run();
        outMap = get_output(pipeline, affinityPolicyName);
        std::vector<FaceInfo> bbox_collection;
        Tensor box_tensor = *(outMap["boxes"].get());
        Tensor score_tensor = *(outMap["scores"].get());
        bounding_boxes_generator(bbox_collection, box_tensor, score_tensor);
        std::vector<FaceInfo> bolt_final_result;
        nms(bbox_collection, bolt_final_result, hard_nms);
        for (unsigned int i = 0; i < bolt_final_result.size(); i++) {
            auto face = bolt_final_result[i];
            cv::Point pt1(face.x1, face.y1);
            cv::Point pt2(face.x2, face.y2);
            cv::rectangle(img, pt1, pt2, cv::Scalar(0, 255, 0), 2);
        }
        cv::imwrite(output_data_path, img);
        break;
    }
    //    LOGI("result saved at=%s",output_data_path.c_str());

    return env->NewStringUTF(output_data_path.c_str());
}
extern "C" JNIEXPORT void JNICALL Java_com_huawei_noah_BoltResult_destroyBolt(
    JNIEnv *env, jobject thiz)
{
    // TODO: implement destroyBolt()
    destroy_priors();
}
