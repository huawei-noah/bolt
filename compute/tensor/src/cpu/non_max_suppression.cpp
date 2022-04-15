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

inline EE qsort_descent(std::vector<BoxRect> &boxes,
    std::vector<I64> &boxindex,
    std::vector<F32> &scores,
    int left,
    int right)
{
    if (boxes.empty() || scores.empty()) {
        return NOT_SUPPORTED;
    }

    int i = left;
    int j = right;
    F32 temp = scores[(left + right) / 2];

    while (i <= j) {
        while (scores[i] > temp) {
            i++;
        }
        while (scores[j] < temp) {
            j--;
        }
        if (i <= j) {
            std::swap(boxes[i], boxes[j]);
            std::swap(scores[i], scores[j]);
            std::swap(boxindex[i], boxindex[j]);
            i++;
            j--;
        }
    }

    if (left < j) {
        qsort_descent(boxes, boxindex, scores, left, j);
    }
    if (i < right) {
        qsort_descent(boxes, boxindex, scores, i, right);
    }

    return SUCCESS;
}

inline F32 intersectionarea(BoxRect a, BoxRect b)
{
    if (a.xmin > b.xmax || a.xmax < b.xmin || a.ymin > b.ymax || a.ymax < b.ymin) {
        return 0.f;
    }
    F32 inter_width = std::min(a.xmax, b.xmax) - std::max(a.xmin, b.xmin);
    F32 inter_height = std::min(a.ymax, b.ymax) - std::max(a.ymin, b.ymin);

    return inter_width * inter_height;
}

inline EE nms_pickedboxes(std::vector<BoxRect> boxes, std::vector<I64> &picked, F32 nms_threshold)
{
    I64 n = boxes.size();

    std::vector<F32> areas(n);
    for (I64 i = 0; i < n; i++) {
        BoxRect box = boxes[i];

        F32 width = box.xmax - box.xmin;
        F32 height = box.ymax - box.ymin;

        areas[i] = width * height;
    }
    for (I64 i = 0; i < n; i++) {
        BoxRect a = boxes[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            BoxRect b = boxes[picked[j]];
            F32 inter_area = intersectionarea(a, b);
            F32 union_area = areas[i] + areas[picked[j]] - inter_area;

            if (inter_area / union_area > nms_threshold) {
                keep = 0;
            }
        }
        if (keep) {
            picked.push_back(i);
        }
    }
    return SUCCESS;
}

template <typename T>
EE non_max_suppression_kernel(std::vector<void *> input,
    T *output,
    U32 spatial_dim,
    U32 num_class,
    U32 max_output_boxes_per_class,
    F32 iou_threshold,
    F32 score_threshold)
{
    T *box = (T *)input[0];
    T *score = (T *)input[1];
    // decode box
    std::vector<std::vector<F32>> boxes;
    boxes.resize(spatial_dim);
    for (U32 i = 0; i < spatial_dim; i++) {
        F32 ymin = std::min<T>(box[i * 4], box[i * 4 + 2]);
        F32 xmin = std::min<T>(box[i * 4 + 1], box[i * 4 + 3]);
        F32 ymax = std::max<T>(box[i * 4], box[i * 4 + 2]);
        F32 xmax = std::max<T>(box[i * 4 + 1], box[i * 4 + 3]);
        std::vector<F32> box_pixel;
        box_pixel.resize(4);
        box_pixel[0] = xmin;
        box_pixel[1] = ymin;
        box_pixel[2] = xmax;
        box_pixel[3] = ymax;
        boxes[i].assign(box_pixel.begin(), box_pixel.end());
    }

    std::vector<BoxInfo> all_boxinfo;
    for (U32 i = 0; i < num_class; i++) {
        std::vector<BoxRect> class_boxrects;
        std::vector<F32> class_boxscores;
        std::vector<I64> class_boxindex;
        for (U32 j = 0; j < spatial_dim; j++) {
            F32 score_pixel = score[i * spatial_dim + j];
            if (score_pixel > score_threshold) {
                std::vector<F32> inbox;
                inbox.assign(boxes[j].begin(), boxes[j].end());
                BoxRect b = {inbox[0], inbox[1], inbox[2], inbox[3], i};
                class_boxrects.push_back(b);
                class_boxindex.push_back(j);
                class_boxscores.push_back(score_pixel);
            }
        }
        // sort boxes and box index
        qsort_descent(class_boxrects, class_boxindex, class_boxscores, 0,
            static_cast<int>(class_boxscores.size() - 1));
        std::vector<I64> picked;
        // apply nms
        nms_pickedboxes(class_boxrects, picked, iou_threshold);
        std::vector<I64> boxindex;
        for (I64 p = 0; p < (I64)picked.size(); p++) {
            I64 picked_box = picked[p];
            boxindex.push_back(class_boxindex[picked_box]);
        }
        if (max_output_boxes_per_class < (U32)boxindex.size()) {
            boxindex.resize(max_output_boxes_per_class);
        }
        for (I64 j = 0; j < (I64)boxindex.size(); j++) {
            BoxInfo bi;
            bi.box_index = boxindex[j];
            bi.label = i;
            all_boxinfo.push_back(bi);
        }
    }
    U32 num_detected = all_boxinfo.size();
    // the first box contains the number of availble boxes in the first element.
    output[0] = num_detected;
    output[1] = output[2] = 0;
    for (U32 i = 0; i < num_detected; i++) {
        BoxInfo bi = all_boxinfo[i];
        // batch_index = 0
        output[(i + 1) * 3] = 0;
        // class_index
        output[(i + 1) * 3 + 1] = bi.label;
        // box_index
        output[(i + 1) * 3 + 2] = bi.box_index;
    }
    return SUCCESS;
}

EE non_max_suppression_cpu(std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    NonMaxSuppressionParamSpec nonMaxSuppressionParamSpec,
    TensorDesc outputDesc,
    void *output)
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
    U32 max_output_boxes_per_class = nonMaxSuppressionParamSpec.max_output_boxes_per_class;
    F32 iou_threshold = nonMaxSuppressionParamSpec.iou_threshold;
    F32 score_threshold = nonMaxSuppressionParamSpec.score_threshold;
    EE ret = SUCCESS;
    switch (idt0) {
#ifdef _USE_FP32
        case DT_F32:
            non_max_suppression_kernel(input, (F32 *)output, spatial_dim, num_class,
                max_output_boxes_per_class, iou_threshold, score_threshold);
            break;
#endif
#ifdef _USE_FP16
        case DT_F16:
            non_max_suppression_kernel(input, (F16 *)output, spatial_dim, num_class,
                max_output_boxes_per_class, iou_threshold, score_threshold);
            break;
#endif
        default:
            ret = NOT_SUPPORTED;
    }
    return ret;
}
