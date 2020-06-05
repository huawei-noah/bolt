// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/tensor_computing_arm.h"
#ifdef _USE_FP32
#include "cpu/arm/fp32/tensor_computing_fp32.h"
#endif
#ifdef _USE_FP16
#include "cpu/arm/fp16/tensor_computing_fp16.h"
#endif


EE detectionoutput_qsort_descent_arm(std::vector<BoxRect>& boxes, std::vector<F32>& scores, int left, int right)
{
    if (boxes.empty() || scores.empty())
        return NOT_SUPPORTED;
    
    int i = left;
    int j = right;
    F32 temp = scores[(left+right) / 2];

    while (i <= j){
        while(scores[i] > temp)
            i++;
        while(scores[j] < temp)
            j--;
        if(i<=j){
            std::swap(boxes[i], boxes[j]);
            std::swap(scores[i], scores[j]);
            i++;
            j--;
        }
    }
    
    if (left < j)
        detectionoutput_qsort_descent_arm(boxes, scores, left, j);
    if (i < right)
        detectionoutput_qsort_descent_arm(boxes, scores, i, right);
    
    return SUCCESS;
}

F32 detectionoutput_intersectionarea_arm(BoxRect a, BoxRect b)
{
    if (a.xmin > b.xmax || a.xmax < b.xmin || a.ymin > b.ymax || a.ymax < b.ymin)
    {
        return 0.f;
    }
    F32 inter_width = std::min(a.xmax, b.xmax) - std::max(a.xmin, b.xmin);
    F32 inter_height = std::min(a.ymax, b.ymax) - std::max(a.ymin, b.ymin);

    return inter_width * inter_height;
}

EE detectionoutput_nms_pickedboxes_arm(std::vector<BoxRect> boxes, std::vector<I64>& picked, F32 nms_threshold)
{
    I64 n = boxes.size();

    std::vector<F32> areas(n);
    for(I64 i = 0; i < n; i++){
        BoxRect box = boxes[i];
          
        F32 width = box.xmax - box.xmin;
        F32 height = box.ymax - box.ymin;
        
        areas[i] = width * height;
    }
    for(I64 i = 0; i < n; i++){

        BoxRect a = boxes[i];
        int keep = 1;
        for(int j = 0; j < (int)picked.size(); j++){
            BoxRect b = boxes[picked[j]];
            F32 inter_area = detectionoutput_intersectionarea_arm(a,b);
            F32 union_area = areas[i] + areas[picked[j]] - inter_area;
        
            if(inter_area / union_area > nms_threshold)
                keep = 0;
        }
        if(keep){
            picked.push_back(i);
        }
    }
    return SUCCESS;
}

EE detectionoutput_arm(std::vector<TensorDesc> inputDesc, std::vector<void*> input, DetectionOutputDesc detectionoutputDesc, TensorDesc outputDesc, void* output)
{
    EE ret = SUCCESS;
    switch (inputDesc[0].dt){
#ifdef _USE_FP32
        case DT_F32: {
            ret = detectionoutput_fp32(inputDesc, input, detectionoutputDesc, outputDesc, (F32*)output);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = detectionoutput_fp16(inputDesc, input, detectionoutputDesc, outputDesc, (F16*)output);
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