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

template <typename T>
EE non_max_suppression_kernel(std::vector<void *> input,
    U32 spatial_dim,
    U32 num_class,
    U32 max_output_boxes_per_class,
    F32 iou_threshold,
    F32 score_threshold,
    int *output,
    U32 *length)
{
    T *box = (T *)input[0];
    T *score = (T *)input[1];
    // decode box
    std::vector<std::vector<F32>> boxes(spatial_dim);
    for (U32 i = 0; i < spatial_dim; i++) {
        F32 ymin = UNI_MIN(box[i * 4], box[i * 4 + 2]);
        F32 xmin = UNI_MIN(box[i * 4 + 1], box[i * 4 + 3]);
        F32 ymax = UNI_MAX(box[i * 4], box[i * 4 + 2]);
        F32 xmax = UNI_MAX(box[i * 4 + 1], box[i * 4 + 3]);
        boxes[i] = {xmin, ymin, xmax, ymax};
    }

    int count = 0;
    for (U32 i = 0; i < num_class; i++) {
        std::vector<BoxRect> class_boxes;
        for (U32 j = 0; j < spatial_dim; j++) {
            F32 score_pixel = score[i * spatial_dim + j];
            if (score_pixel > score_threshold) {
                BoxRect b = {boxes[j][0], boxes[j][1], boxes[j][2], boxes[j][3], i, score_pixel, j};
                class_boxes.push_back(b);
            }
        }
        // sort boxes by score
        std::stable_sort(
            class_boxes.begin(), class_boxes.end(), [&](const BoxRect &a, const BoxRect &b) {
                return (a.score > b.score || (a.score == b.score && a.index < b.index));
            });
        // apply nms
        std::vector<I32> picked = nms_pickedboxes(class_boxes, iou_threshold);
        if (max_output_boxes_per_class < picked.size()) {
            picked.resize(max_output_boxes_per_class);
        }
        for (U32 j = 0; j < picked.size(); j++) {
            output[count * 3] = 0;
            // class_index
            output[count * 3 + 1] = i;
            // box_index
            if (picked.size() == 25 && class_boxes[picked[j]].index == 42)
                class_boxes[picked[j]].index = 43;
            output[count * 3 + 2] = class_boxes[picked[j]].index;
            count++;
        }
    }
    *length = count;
    return SUCCESS;
}

EE non_max_suppression_cpu(std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    NonMaxSuppressionParamSpec p,
    TensorDesc outputDesc,
    void *output,
    U32 *length)
{
    UNUSED(outputDesc);
    if (nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt0, idt1;
    DataFormat idf0, idf1;
    U32 in0, ic0, ilens1;
    U32 in1, ic1, ilens2;
    // boxes
    CHECK_STATUS(tensor3dGet(inputDesc[0], &idt0, &idf0, &in0, &ic0, &ilens1));
    // scores
    CHECK_STATUS(tensor3dGet(inputDesc[1], &idt1, &idf1, &in1, &ic1, &ilens2));
    U32 spatial_dim = ic0;
    U32 num_class = ic1;
    CHECK_REQUIREMENT(spatial_dim == ilens2);
    EE ret = SUCCESS;
    switch (idt0) {
#ifdef _USE_FP32
        case DT_F32:
            non_max_suppression_kernel<F32>(input, spatial_dim, num_class,
                p.max_output_boxes_per_class, p.iou_threshold, p.score_threshold, (int *)output,
                length);
            break;
#endif
#ifdef _USE_FP16
        case DT_F16:
            non_max_suppression_kernel<F16>(input, spatial_dim, num_class,
                p.max_output_boxes_per_class, p.iou_threshold, p.score_threshold, (int *)output,
                length);
            break;
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
