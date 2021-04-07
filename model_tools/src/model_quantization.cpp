// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <map>
#include <sstream>
#include <fstream>
#include "model_quantization.h"
#include "model_common.h"

std::vector<std::string> SplitScale(const std::string &s, char delim)
{
    std::vector<std::string> res;
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, delim)) {
        res.push_back(item);
    }

    return res;
}

void add_scale_from_file(ModelSpec *ms, const char *scaleFile)
{
    std::fstream file(std::string(scaleFile), std::ios::in);
    CHECK_REQUIREMENT(file && file.is_open());
    std::map<std::string, std::vector<F32>> scaleMap;
    std::string line;
    UNI_DEBUG_LOG("Scale Table is : \n");
    while (std::getline(file, line)) {
        auto res = SplitScale(line, ' ');
        CHECK_REQUIREMENT(res.size() == 2);
        std::string tensorName = res[0];
        std::vector<F32> quantScale;
        quantScale.push_back(atof(res[1].c_str()));
        scaleMap[tensorName] = quantScale;
        UNI_DEBUG_LOG("Tensor[%s] %f\n", tensorName.c_str(), quantScale[0]);
    }
    file.close();
    for (I32 i = 0; i < (*ms).num_operator_specs; i++) {
        if (isDeprecatedOp((*ms).ops[i].type)) {
            continue;
        }
        if ((*ms).ops[i].num_quant_feature == 1 && (*ms).ops[i].feature_scale[0].scale[0] == 0) {
            continue;
        }
        std::vector<std::vector<F32>> scales;
        for (U32 j = 0; j < (*ms).ops[i].num_inputs; j++) {
            auto it = scaleMap.find((*ms).ops[i].input_tensors_name[j]);
            std::vector<F32> inputScale;
            if (it != scaleMap.end()) {
                inputScale.push_back(127.0f / scaleMap[(*ms).ops[i].input_tensors_name[j]][0]);
            } else {
                inputScale.push_back(-1);
            }
            scales.push_back(inputScale);
        }
        for (U32 j = 0; j < (*ms).ops[i].num_outputs; j++) {
            auto it = scaleMap.find((*ms).ops[i].output_tensors_name[j]);
            std::vector<F32> outputScale;
            if ((*ms).ops[i].num_quant_feature == 1 && -2 == (*ms).ops[i].feature_scale[0].scale[0]) {
                outputScale.push_back(-2);
            } else if (it != scaleMap.end()) {
                outputScale.push_back(127.0f / scaleMap[(*ms).ops[i].output_tensors_name[j]][0]);
            } else {
                outputScale.push_back(-1);
            }
            scales.push_back(outputScale);
        }
        // Store scales into result model
        if (nullptr != (*ms).ops[i].feature_scale) {  // Could be labelled with -2
            for (U32 k = 0; k < (*ms).ops[i].num_quant_feature; k++) {
                if (nullptr != (*ms).ops[i].feature_scale[k].scale) {
                    delete (*ms).ops[i].feature_scale[k].scale;
                }
            }
            delete (*ms).ops[i].feature_scale;
        }

        (*ms).ops[i].num_quant_feature = scales.size();
        (*ms).ops[i].feature_scale = (QuantSpec *)mt_new_storage(scales.size() * sizeof(QuantSpec));

        for (U32 k = 0; k < scales.size(); k++) {
            (*ms).ops[i].feature_scale[k].num_scale = scales[k].size();
            U32 scaleBytes = scales[k].size() * sizeof(F32);
            (*ms).ops[i].feature_scale[k].scale = (F32 *)mt_new_storage(scaleBytes);
            memcpy((*ms).ops[i].feature_scale[k].scale, scales[k].data(), scaleBytes);
        }
    }
}
