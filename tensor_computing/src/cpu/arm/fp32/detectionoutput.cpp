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
#include "cpu/arm/fp32/tensor_computing_fp32.h"

EE detectionoutput_fp32(std::vector<TensorDesc> inputDesc, std::vector<void*> input, DetectionOutputDesc detectionoutputDesc, TensorDesc outputDesc, F32* output)
{
    UNUSED(outputDesc);
    if (inputDesc.size() != 3) {
        CHECK_STATUS(NOT_MATCH);
    }
    F32* location = (F32*)input[0];
    F32* confidence = (F32*)input[1];
    F32* priorbox = (F32*)input[2];

    U32 ilens2 = inputDesc[2].dims[0];
    U32 num_total_priorbox = ilens2 / 4;
    U32 num_class = detectionoutputDesc.num_class;
    F32 nms_threshold = detectionoutputDesc.nms_threshold;
    U32 nms_top_k = detectionoutputDesc.nms_top_k;
    U32 keep_top_k = detectionoutputDesc.keep_top_k;
    F32 confidence_threshold = detectionoutputDesc.confidence_threshold;

    std::vector<std::vector<F32>> boxes;
    boxes.resize(num_total_priorbox);
    F32* variance = priorbox + ilens2;
    // decode priorbox
    for(U32 i = 0 ;  i < num_total_priorbox ; i++){
        F32* loc = location + i * 4;
        F32* pb = priorbox + i * 4;
        F32* var = variance + i * 4;
   
        F32 pb_w = pb[2] - pb[0];
        F32 pb_h = pb[3] - pb[1];
        F32 pb_cx = (pb[0] + pb[2]) * 0.5f;
        F32 pb_cy = (pb[1] + pb[3]) * 0.5f;

        F32 box_cx = var[0] * loc[0] * pb_w + pb_cx;
        F32 box_cy = var[1] * loc[1] * pb_h + pb_cy;
        F32 box_w = static_cast<F32>(exp(var[2] * loc[2]) * pb_w);
        F32 box_h = static_cast<F32>(exp(var[3] * loc[3]) * pb_h);
        
        std::vector<F32> box;
        box.resize(4);
        box[0] = box_cx - box_w * 0.5f;
        box[1] = box_cy - box_h * 0.5f;
        box[2] = box_cx + box_w * 0.5f;
        box[3] = box_cy + box_h * 0.5f;
        // give box to boxes
        boxes[i].assign(box.begin(),box.end());
    }

    std::vector<std::vector<BoxRect>> allclass_boxrects;
    std::vector<std::vector<F32>> allclass_boxscores;
    allclass_boxrects.resize(num_class);
    allclass_boxscores.resize(num_class);

    for(U32 i = 1; i < num_class; i++){
        std::vector<BoxRect> class_boxrects;
        std::vector<F32> class_boxscores;
        for(U32 j = 0; j < num_total_priorbox; j++){
            F32 score = confidence[j * num_class + i];
            if (score > confidence_threshold)
            {
                std::vector<F32> inbox;
                inbox.assign(boxes[j].begin(),boxes[j].end());
                BoxRect b = { inbox[0], inbox[1], inbox[2], inbox[3], i };
                class_boxrects.push_back(b);
                class_boxscores.push_back(score);
            }
        }
        //sort the boxes with scores
        detectionoutput_qsort_descent_arm(class_boxrects, class_boxscores, 0, static_cast<int>(class_boxscores.size()-1));

        if(nms_top_k < (U32)class_boxrects.size()){
            class_boxrects.resize(nms_top_k);
            class_boxscores.resize(nms_top_k);
        }
        //apply nms
        std::vector<I64> picked;
        detectionoutput_nms_pickedboxes_arm(class_boxrects, picked, nms_threshold);

        for(I64 j = 0; j < (I64)picked.size(); j++)
        {
            I64 picked_box = picked[j];
            allclass_boxrects[i].push_back(class_boxrects[picked_box]);
            allclass_boxscores[i].push_back(class_boxscores[picked_box]);
        }
    }

    std::vector<BoxRect> boxrects;
    std::vector<F32> boxscores;

    for (U32 i = 1; i < num_class ; i++)
    {
        boxrects.insert(boxrects.end(), allclass_boxrects[i].begin(), allclass_boxrects[i].end());
        boxscores.insert(boxscores.end(), allclass_boxscores[i].begin(), allclass_boxscores[i].end());
    }

    detectionoutput_qsort_descent_arm(boxrects, boxscores, 0, static_cast<int>(boxscores.size()-1));

    if (keep_top_k < (U32)boxrects.size())
    {
        boxrects.resize(keep_top_k);
        boxscores.resize(keep_top_k);
    }

    U32 num_detected = static_cast<U32>(boxrects.size());
    if (num_detected == 0)
        return SUCCESS;

    // the first box contains the number of availble boxes
    output[0] = num_detected;
    output[1] = output[2] = output[3] = output[4] = output[5] = 0;

    for(U32 i = 0; i < num_detected ; i++){
        BoxRect b = boxrects[i];
        F32 score = boxscores[i];

        output[(i+1)*6] = b.label;
        output[(i+1)*6+1] = score;
        output[(i+1)*6+2] = b.xmin;
        output[(i+1)*6+3] = b.ymin;
        output[(i+1)*6+4] = b.xmax;
        output[(i+1)*6+5] = b.ymax;
    }
    return SUCCESS;
}