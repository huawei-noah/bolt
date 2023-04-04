// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/tensor_computing_cpu.h"
#include "cpu/non_max_suppression.h"
#include "tensor_transpose.h"

template <typename T>
EE yolov3detectionoutput(std::vector<void *> input,
    T *output,
    std::vector<TensorDesc> inputDesc,
    Yolov3DetectionOutputParamSpec yolov3DetectionOutputParamSpec,
    Arch arch)
{
    U32 num_class = yolov3DetectionOutputParamSpec.num_class;
    U32 num_box = yolov3DetectionOutputParamSpec.num_box;
    F32 confidence_threshold = yolov3DetectionOutputParamSpec.confidence_threshold;
    F32 nms_threshold = yolov3DetectionOutputParamSpec.nms_threshold;
    std::vector<F32> biases;
    for (int i = 0; i < 18; i++) {
        if (yolov3DetectionOutputParamSpec.biases[i] == 0) {
            break;
        }
        biases.push_back(yolov3DetectionOutputParamSpec.biases[i]);
    }
    std::vector<U32> anchors_scale;
    for (int i = 0; i < 3; i++) {
        if (yolov3DetectionOutputParamSpec.anchors_scale[i] == 0) {
            break;
        }
        anchors_scale.push_back(yolov3DetectionOutputParamSpec.anchors_scale[i]);
    }
    std::vector<U32> mask;
    for (int i = 0; i < (int)(yolov3DetectionOutputParamSpec.mask_group_num * 3); i++) {
        mask.push_back(yolov3DetectionOutputParamSpec.mask[i]);
    }

    std::vector<BoxRect> all_boxrects;
    I64 input_size = inputDesc.size();
    U32 info_per_box = 4 + 1 + num_class;
    ActivationParamSpec activationdesc_sigmoid;
    activationdesc_sigmoid.mode = ACTIVATION_SIGMOID;
    TensorDesc tmpDesc = tensor1d(inputDesc[0].dt, 1);
    for (I64 i = 0; i < input_size; i++) {
        T *in = (T *)input[i];
        CHECK_REQUIREMENT(inputDesc[i].df == DF_NCHWC8 || inputDesc[i].df == DF_NCHW);
        if (inputDesc[i].df == DF_NCHWC8) {
            T *tmp = (T *)malloc(tensorNumBytes(inputDesc[0]));
            UNI_MEMCPY(tmp, in, tensorNumBytes(inputDesc[0]));
            CHECK_STATUS(transformToNCHW(inputDesc[0], tmp, inputDesc[0], in));
            free(tmp);
        }
        std::vector<std::vector<BoxRect>> allbox_boxrects(num_box);

        U32 w = inputDesc[i].dims[0];
        U32 h = inputDesc[i].dims[1];
        U32 net_w = (U32)(anchors_scale[i] * w);
        U32 net_h = (U32)(anchors_scale[i] * h);
        I64 mask_offset = i * num_box;
        U32 hw_stride = w * h;
        U32 idx = 0;

        for (U32 b = 0; b < num_box; b++) {
            U32 biases_index = mask[b + mask_offset];
            F32 bias_w = biases[biases_index * 2];
            F32 bias_h = biases[biases_index * 2 + 1];
            idx = hw_stride * b * info_per_box;
            for (U32 nh = 0; nh < h; nh++) {
                for (U32 nw = 0; nw < w; nw++) {
                    T box_score = 0;
                    CHECK_STATUS(activation_cpu(tmpDesc, &in[idx + 4 * hw_stride],
                        activationdesc_sigmoid, tmpDesc, &box_score, nullptr, arch));
                    U32 label = 0;
                    T class_score_max = in[idx + 5 * hw_stride];
                    T class_score = 0;
                    for (U32 c = 1; c < num_class; c++) {
                        class_score = in[idx + (5 + c) * hw_stride];
                        if (class_score > class_score_max) {
                            label = c;
                            class_score_max = class_score;
                        }
                    }
                    CHECK_STATUS(activation_cpu(tmpDesc, &class_score_max, activationdesc_sigmoid,
                        tmpDesc, &class_score, nullptr, arch));
                    F32 score_conf = static_cast<F32>(box_score * class_score);
                    T cx, cy;
                    cx = cy = 0;
                    if (score_conf >= confidence_threshold) {
                        CHECK_STATUS(activation_cpu(
                            tmpDesc, &in[idx], activationdesc_sigmoid, tmpDesc, &cx, nullptr, arch));
                        F32 box_cx = static_cast<F32>((nw + cx) / w);
                        CHECK_STATUS(activation_cpu(tmpDesc, &in[idx + 1 * hw_stride],
                            activationdesc_sigmoid, tmpDesc, &cy, nullptr, arch));
                        F32 box_cy = static_cast<F32>((nh + cy) / h);
                        F32 box_w = static_cast<F32>(exp(in[idx + 2 * hw_stride]) * bias_w / net_w);
                        F32 box_h = static_cast<F32>(exp(in[idx + 3 * hw_stride]) * bias_h / net_h);

                        F32 box_xmin = box_cx - box_w * 0.5;
                        F32 box_ymin = box_cy - box_h * 0.5;
                        F32 box_xmax = box_cx + box_w * 0.5;
                        F32 box_ymax = box_cy + box_h * 0.5;
                        BoxRect box = {
                            box_xmin, box_ymin, box_xmax, box_ymax, label, score_conf, INT_MAX};
                        allbox_boxrects[b].push_back(box);
                    }
                    idx++;
                }
            }
        }

        for (U32 b = 0; b < num_box; b++) {
            all_boxrects.insert(
                all_boxrects.end(), allbox_boxrects[b].begin(), allbox_boxrects[b].end());
        }
    }
    // sort boxes
    std::stable_sort(all_boxrects.begin(), all_boxrects.end(),
        [&](const BoxRect &a, const BoxRect &b) { return (a.score > b.score); });
    // apply nms
    std::vector<I32> picked = nms_pickedboxes(all_boxrects, nms_threshold);

    std::vector<BoxRect> boxrects;
    for (U32 p = 0; p < picked.size(); p++) {
        I64 picked_box = picked[p];
        boxrects.push_back(all_boxrects[picked_box]);
    }

    U32 num_detected = boxrects.size();
    // the first box contains the number of availble boxes
    output[0] = num_detected;
    output[1] = output[2] = output[3] = output[4] = output[5] = 0;
    for (U32 i = 0; i < num_detected; i++) {
        BoxRect b = boxrects[i];
        output[(i + 1) * 6] = b.label + 1;
        output[(i + 1) * 6 + 1] = b.score;
        output[(i + 1) * 6 + 2] = b.xmin;
        output[(i + 1) * 6 + 3] = b.ymin;
        output[(i + 1) * 6 + 4] = b.xmax;
        output[(i + 1) * 6 + 5] = b.ymax;
    }
    return SUCCESS;
}

EE yolov3detectionoutput_cpu(std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    Yolov3DetectionOutputParamSpec yolov3DetectionOutputParamSpec,
    TensorDesc outputDesc,
    void *output,
    Arch arch)
{
    UNUSED(outputDesc);
    if (nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    EE ret = SUCCESS;
    switch (inputDesc[0].dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = yolov3detectionoutput(
                input, (F32 *)output, inputDesc, yolov3DetectionOutputParamSpec, arch);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = yolov3detectionoutput(
                input, (F16 *)output, inputDesc, yolov3DetectionOutputParamSpec, arch);
            break;
        }
#endif
        default: {
            ret = NOT_SUPPORTED;
            break;
        }
    }
    return ret;
}
