// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#ifndef _H_ULTRA_FACE
#define _H_ULTRA_FACE

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include "inference.hpp"
#include "profiling.h"

#define num_featuremap 4
#define hard_nms 1
#define blending_nms 2

typedef struct FaceInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
} FaceInfo;

void print_face_info(FaceInfo face_info)
{
    std::cout << "x1: " << face_info.x1 << std::endl;
    std::cout << "y1: " << face_info.y1 << std::endl;
    std::cout << "x2: " << face_info.x2 << std::endl;
    std::cout << "y2: " << face_info.y2 << std::endl;
    std::cout << "score: " << face_info.score << "\n\n";
}

int image_w;
int image_h;
int in_w;
int in_h;
int num_anchors;
float score_threshold;
float iou_threshold;

const float mean_vals[3] = {127, 127, 127};
const float norm_vals[3] = {1.0 / 128, 1.0 / 128, 1.0 / 128};
const float center_variance = 0.1;
const float size_variance = 0.2;
const std::vector<std::vector<float>> min_boxes = {
    {10.0f, 16.0f, 24.0f}, {32.0f, 48.0f}, {64.0f, 96.0f}, {128.0f, 192.0f, 256.0f}};
const std::vector<float> strides = {8.0, 16.0, 32.0, 64.0};
std::vector<std::vector<float>> featuremap_size;
std::vector<std::vector<float>> shrinkage_size;
std::vector<int> w_h_list;
std::vector<std::vector<float>> priors = {};

inline float clip(float x, float y)
{
    float ret = (x < 0 ? 0 : (x > y ? y : x));
    return ret;
}

inline void prior_boxes_generator(
    int input_width, int input_length, float score_threshold, float iou_threshold)
{
    in_w = input_width;
    in_h = input_length;
    w_h_list = {in_w, in_h};
    for (auto size : w_h_list) {
        std::vector<float> fm_item;
        for (float stride : strides) {
            fm_item.push_back(ceil(size / stride));
        }
        featuremap_size.push_back(fm_item);
    }
    int row_num = featuremap_size.size();
    int col_num = featuremap_size[0].size();

    for (auto size : w_h_list) {
        shrinkage_size.push_back(strides);
    }
    row_num = shrinkage_size.size();
    col_num = shrinkage_size[0].size();

    /*generate prior anchors*/
    for (int index = 0; index < num_featuremap; index++) {
        float scale_w = in_w / shrinkage_size[0][index];
        float scale_h = in_h / shrinkage_size[1][index];
        for (int j = 0; j < featuremap_size[1][index]; j++) {
            for (int i = 0; i < featuremap_size[0][index]; i++) {
                float x_center = (i + 0.5) / scale_w;
                float y_center = (j + 0.5) / scale_h;
                for (float k : min_boxes[index]) {
                    float w = k / in_w;
                    float h = k / in_h;
                    priors.push_back({clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1)});
                }
            }
        }
    }
    num_anchors = priors.size();
}

inline void bounding_boxes_generator(
    std::vector<FaceInfo> &bbox_collection, Tensor box_tensor, Tensor score_tensor)
{
    float *box_ptr = (float *)(((CpuMemory *)box_tensor.get_memory())->get_ptr());
    float *score_ptr = (float *)(((CpuMemory *)score_tensor.get_memory())->get_ptr());

    for (int i = 0; i < num_anchors; i++) {
        if (score_ptr[i * 2 + 1] > 0.7) {
            FaceInfo rects;
            float x_center = box_ptr[i * 4] * center_variance * priors[i][2] + priors[i][0];
            float y_center = box_ptr[i * 4 + 1] * center_variance * priors[i][3] + priors[i][1];
            float w = exp(box_ptr[i * 4 + 2] * size_variance) * priors[i][2];
            float h = exp(box_ptr[i * 4 + 3] * size_variance) * priors[i][3];
            rects.x1 = clip(x_center - w / 2.0, 1) * image_w;
            rects.y1 = clip(y_center - h / 2.0, 1) * image_h;
            rects.x2 = clip(x_center + w / 2.0, 1) * image_w;
            rects.y2 = clip(y_center + h / 2.0, 1) * image_h;
            rects.score = clip(score_ptr[i * 2 + 1], 1);
            bbox_collection.push_back(rects);
        }
    }
}

inline int nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, int type)
{
    std::sort(input.begin(), input.end(),
        [](const FaceInfo &a, const FaceInfo &b) { return a.score > b.score; });
    int box_num = input.size();
    std::vector<int> merged(box_num, 0);
    for (int i = 0; i < box_num; i++) {
        if (merged[i]) {
            continue;
        }
        std::vector<FaceInfo> buf;
        buf.push_back(input[i]);
        merged[i] = 1;
        float h0 = input[i].y2 - input[i].y1 + 1;
        float w0 = input[i].x2 - input[i].x1 + 1;
        float area0 = h0 * w0;
        for (int j = i + 1; j < box_num; j++) {
            if (merged[j]) {
                continue;
            }
            float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
            float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;
            float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
            float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;
            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;
            if (inner_h <= 0 || inner_w <= 0) {
                continue;
            }
            float inner_area = inner_h * inner_w;
            float h1 = input[j].y2 - input[j].y1 + 1;
            float w1 = input[j].x2 - input[j].x1 + 1;
            float area1 = h1 * w1;
            float score;
            score = inner_area / (area0 + area1 - inner_area);
            if (score > 0.3) {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }
        switch (type) {
            case hard_nms: {
                output.push_back(buf[0]);
                break;
            }
            case blending_nms: {
                float total = 0;
                for (unsigned int i = 0; i < buf.size(); i++) {
                    total += exp(buf[i].score);
                }
                FaceInfo rects;
                UNI_MEMSET(&rects, 0, sizeof(rects));
                for (unsigned int i = 0; i < buf.size(); i++) {
                    float rate = exp(buf[i].score) / total;
                    rects.x1 += buf[i].x1 * rate;
                    rects.y1 += buf[i].y1 * rate;
                    rects.x2 += buf[i].x2 * rate;
                    rects.y2 += buf[i].y2 * rate;
                    rects.score += buf[i].score * rate;
                }
                output.push_back(rects);
                break;
            }
            default: {
                printf("wrong type of nms.");
                return 1;
            }
        }
    }
    return 0;
}
#endif
