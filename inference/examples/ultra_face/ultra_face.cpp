// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ultra_face.h"
#include <getopt.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
using namespace cv;

char *modelPath = (char *)"";
std::string inputData = "";
char *affinityPolicyName = (char *)"CPU_AFFINITY_HIGH_PERFORMANCE";
char *algorithmMapPath = (char *)"";

void print_ultraface_usage()
{
    std::cout << "ultra_face usage: (<> must be filled in with exact value)\n"
                 "./ultra_face -m <boltModelPath> -i <inputDataPath>\n"
                 "\nParameter description:\n"
                 "1. -m <boltModelPath>: The path where .bolt is stored.\n"
                 "2. -i [inputDataPath]: The input video data(avi) absolute path.\n"
                 "Example: ./ultra_face -m ./ultra_face_fp32.bolt -i ./face_detection_sample.avi\n"
                 "The output video is : ./face_detection_sample_bolt.avi"
              << std::endl;
}

int parse_options(int argc, char *argv[])
{
    std::cout << "\nPlease enter this command './benchmark --help' to get more usage "
                 "information.\n";
    std::vector<std::string> lineArgs(argv, argv + argc);
    for (std::string arg : lineArgs) {
        if (arg == "--help" || arg == "-help" || arg == "--h" || arg == "-h") {
            print_ultraface_usage();
            return 1;
        }
    }

    int option;
    const char *optionstring = "m:i:";
    while ((option = getopt(argc, argv, optionstring)) != -1) {
        switch (option) {
            case 'm':
                std::cout << "option is -m <boltModelPath>, value is: " << optarg << std::endl;
                modelPath = optarg;
                break;
            case 'i':
                std::cout << "option is -i [inputDataPath], value is: " << optarg << std::endl;
                inputData = std::string(optarg);
                break;
            default:
                std::cerr << "Input option gets error, please check the params meticulously.\n";
                print_ultraface_usage();
                return 1;
        }
    }
    return 0;
}

std::map<std::string, std::shared_ptr<Tensor>> get_output(
    std::shared_ptr<CNN> pipeline, std::string affinity)
{
    std::map<std::string, std::shared_ptr<Tensor>> outMap = pipeline->get_output();
    if (affinity == "GPU") {
#ifdef _USE_GPU
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

int main(int argc, char *argv[])
{
    if (0 != parse_options(argc, argv)) {
        return 1;
    }

    prior_boxes_generator(320, 240, 0.7, 0.3);  // debug check the size of prior
    int last_gang_index = inputData.find_last_of('/');
    int last_dot_index = inputData.find_last_of('.');
    std::string prefix_str = inputData.substr(0, last_gang_index + 1);
    std::string video_name =
        inputData.substr(last_gang_index + 1, last_dot_index - last_gang_index - 1);
    std::string suffix_str = inputData.substr(last_dot_index, inputData.length());
    std::string output_video_path = prefix_str + "bolt_" + video_name + suffix_str;

    VideoCapture cap(inputData);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open the video file. \n";
        return 1;
    } else {
        std::cout << "Successfully open the video! \n\n";
    }
    int frame_width = cap.get(3);
    int frame_height = cap.get(4);
    int frame_rate = cap.get(5);
    VideoWriter video(output_video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), frame_rate,
        Size(frame_width, frame_height));

    // deal with the first frame and set up the global variables
    cv::Mat img;
    cap >> img;
    if (img.empty()) {
        std::cout << "ERROR: video is empty(), please check the input video.\n";
        return 0;
    }
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
        video.write(img);
        cap >> img;
        if (img.empty()) {
            break;
        }
    }
    std::cout << "result saved at " << output_video_path << std::endl;
    return 0;
}
