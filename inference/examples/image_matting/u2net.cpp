// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <iostream>
#include <string>
#include "inference.hpp"
#include "data_loader.hpp"
#include "result_format.hpp"
#include "profiling.h"
#include "parse_command.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

char *modelPath = (char *)"";
std::string inputData = "";
char *affinityPolicyName = (char *)"CPU_AFFINITY_HIGH_PERFORMACE";
char *algorithmMapPath = (char *)"";

void print_image_matting_usage()
{
    std::cout << "image_matting usage: (<> must be filled in with exact value)\n"
                 "./ultra_face -m <boltModelPath> -i <inputDataPath>\n"
                 "\nParameter description:\n"
                 "1. -m <boltModelPath>: The path where .bolt is stored.\n"
                 "2. -i [inputDataPath]: The input image absolute path.\n"
                 "Example: ./image_matting -m ./u2net_fp32.bolt -i ./test_pic.jpg\n"
                 "The output pic is : ./bolt_test_pic.jpg\n"
                 "Note: Only support single picture matting by now.\n"
              << std::endl;
}

int parse_options(int argc, char *argv[])
{
    std::cout << "\nPlease enter this command './benchmark --help' to get more usage "
                 "information.\n";
    std::vector<std::string> lineArgs(argv, argv + argc);
    for (std::string arg : lineArgs) {
        if (arg == "--help" || arg == "-help" || arg == "--h" || arg == "-h") {
            print_image_matting_usage();
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
                print_image_matting_usage();
                return 1;
        }
    }
    return 0;
}

std::shared_ptr<U8> preprocess(cv::Mat image,
    int width_model,
    int height_model,
    cv::Ptr<cv::DenseOpticalFlow> flow_algorithm,
    int flag = 5)
{
    std::map<std::string, std::shared_ptr<U8>> model_tensors_input;
    std::vector<float> vec_flow;
    int appending_channels = 0;
    if (flag == 5) {
        cv::Mat gray, flow;
        cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
        flow_algorithm->calc(gray, gray, flow);
        cv::resize(flow, flow, cv::Size(width_model, height_model));
        vec_flow.assign((float *)flow.datastart, (float *)flow.dataend);
        appending_channels = 2;  // decide the memory capacity to append
    }

    image.convertTo(image, CV_32F);  // attention to the data precision
    cv::resize(image, image, cv::Size(width_model, height_model));
    std::vector<float> vec_original;
    vec_original.assign((float *)image.datastart, (float *)image.dataend);
    float max = *std::max_element(vec_original.begin(), vec_original.end());
    std::shared_ptr<U8> input_ptr(new U8[image.size().height * image.size().width *
        (image.channels() + appending_channels) * sizeof(float)]);
    float *vec_transpose = (float *)(input_ptr.get());
    int iter_index = 0;
    std::vector<int> channel_order = {2, 0, 1};
    for (int k = 0; k < image.channels(); k++) {
        int i = channel_order[k];
        float p1 = 0, p2 = 1;
        if (i == 0) {
            p1 = 0.485;
            p2 = 0.229;
        } else if (i == 1) {
            p1 = 0.456;
            p2 = 0.229;
        } else if (i == 2) {
            p1 = 0.406;
            p2 = 0.225;
        }
        // channel-wise normalization
        for (unsigned int j = 0; j < vec_original.size() / image.channels(); j++) {
            vec_transpose[iter_index] = ((vec_original[j * image.channels() + i] / max - p1)) / p2;
            iter_index++;
        }
    }
    if (appending_channels != 0) {
        UNI_MEMCPY(&(vec_transpose[iter_index]), &(vec_flow[0]), vec_flow.size() * sizeof(float));
    }
    return input_ptr;
}

void generate_output_path(std::string input_path, std::string &output_path)
{
    int last_gang_index = input_path.find_last_of('/');
    int last_dot_index = input_path.find_last_of('.');
    std::string prefix_str = input_path.substr(0, last_gang_index + 1);
    std::string video_name =
        input_path.substr(last_gang_index + 1, last_dot_index - last_gang_index - 1);
    std::string suffix_str = input_path.substr(last_dot_index, input_path.length());
    output_path = prefix_str + "bolt_" + video_name + suffix_str;
}

int main(int argc, char *argv[])
{
    UNI_TIME_INIT
    if (0 != arse_options(argc, argv)) {
        return 1;
    }

    auto nn = createPipeline(affinityPolicyName, modelPath, algorithmMapPath);
    std::map<std::string, std::shared_ptr<Tensor>> inMap = nn->get_input();
    TensorDesc nnDesc = (*(inMap.begin()->second)).get_desc();
    int width_model = nnDesc.dims[0];
    int height_model = nnDesc.dims[1];

    double totalTime = 0;
    double max_time = -DBL_MAX;
    double min_time = DBL_MAX;

    cv::Ptr<cv::DenseOpticalFlow> flow_algorithm =
        cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
    cv::Mat img_bg(width_model, height_model, CV_32FC3, cv::Scalar(12, 198, 150));

    // load image
    cv::Mat image = cv::imread(inputData);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    // image.convertTo(image, CV_32F);
    int height_original = image.size().height;
    int width_original = image.size().width;

    double timeBegin = ut_time_ms();

    // prepare input for model
    std::map<std::string, std::shared_ptr<U8>> model_tensors_input;
    model_tensors_input[nn->get_input_desc().begin()->first] =
        preprocess(image, width_model, height_model, flow_algorithm, 5);
    nn->set_input_by_assign(model_tensors_input);

    double t1 = ut_time_ms();

    // model infer
    nn->run();
    double t2 = ut_time_ms();
    image.convertTo(image, CV_32F);

    // combine after infer
    std::map<std::string, std::shared_ptr<Tensor>> outMap = nn->get_output();
    Tensor result = *(outMap.begin()->second);
    float *vec_result = (float *)(((CpuMemory *)result.get_memory())->get_ptr());
    cv::Mat mask(width_model, height_model, CV_32F, vec_result);

    cv::resize(mask, mask, cv::Size(width_original, height_original));
    cv::Mat repeat[] = {mask, mask, mask};
    cv::merge(repeat, 3, mask);

    // here
    cv::Mat bg;
    cv::resize(img_bg, bg, cv::Size(width_original, height_original));
    cv::Mat part_1 = image.mul(mask);
    cv::Mat part_2 = bg.mul(1 - mask);
    part_1.convertTo(part_1, CV_8U);
    part_2.convertTo(part_2, CV_8U);
    cv::Mat output = part_1 + part_2;

    double timeEnd = ut_time_ms();
    totalTime += (timeEnd - timeBegin);

    std::string file_output = "";
    generate_output_path(inputData, file_output);
    output.convertTo(output, CV_8U);
    cv::cvtColor(output, output, cv::COLOR_RGB2BGR);
    cv::imwrite(file_output, output);
    std::cout << "Succeed! Result saved at: " << file_output << "\n\n";
    UNI_TIME_STATISTICS
    UNI_CI_LOG("avg_time:%fms/image\n", 1.0 * totalTime);
    return 0;
}
