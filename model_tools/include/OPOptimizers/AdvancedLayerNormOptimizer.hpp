// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_ADVANCEDLAYERNORMOPTIMIZER
#define _H_ADVANCEDLAYERNORMOPTIMIZER

#include "OPOptimizer.hpp"
#include <map>

class AdvancedLayerNormOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        int totalOptimizedCount = 0;
        // pre-defined
        std::map<OperatorType, int> transposeMode = {
            {OT_Reduction, 2}, {OT_Transpose, 3}, {OT_Eltwise, 5}, {OT_Power, 3}, {OT_Scale, 1}};
        std::map<OperatorType, int> nonTransposeMode = {
            {OT_Reduction, 2}, {OT_Eltwise, 5}, {OT_Power, 3}, {OT_Scale, 1}};

        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (OT_Reduction == spec->ops[i].type) {
                std::map<OperatorType, int> tmpMap;
                int curIndex = i;
                int addIndex = -1;
                int mulIndex = -1;
                int subIndex = -1;
                int transIndex = -1;  // new

                std::string mulWeightName = "";
                std::string subWeightName = "";
                int groupLastIndex = -1;
                int opCount = 0;
                for (int j = curIndex; j < spec->num_operator_specs; j++) {
                    if (spec->ops[j].type != OT_None) {
                        opCount++;
                    } else {
                        continue;
                    }
                    if (tmpMap.find(spec->ops[j].type) == tmpMap.end()) {
                        tmpMap[spec->ops[j].type] = 1;
                    } else {
                        tmpMap[spec->ops[j].type] += 1;
                    }

                    if (OT_Power == spec->ops[j].type) {
                        auto cur_ps = spec->ops[j].ps.power_spec;
                        if (1 == cur_ps.scale && 1 == cur_ps.power && cur_ps.shift != 0) {
                            addIndex = j;
                        }
                    }
                    if (OT_Scale == spec->ops[j].type) {
                        mulIndex = j;
                    }
                    if (OT_Eltwise == spec->ops[j].type &&
                        ELTWISE_SUB == spec->ops[j].ps.eltwise_spec.mode) {
                        subIndex = j;
                    }
                    if (tmpMap == nonTransposeMode && opCount == 11) {
                        groupLastIndex = j;
                        break;
                    }
                    if (OT_Transpose == spec->ops[j].type) {  // new
                        transIndex = j;
                    }
                    if (opCount == 14) {
                        groupLastIndex = j;
                        break;
                    }
                }

                if (addIndex < 0 || mulIndex < 0 || subIndex < 0) {
                    continue;
                } else {
                    mulWeightName = std::string(spec->ops[mulIndex].name);
                    subWeightName = std::string(spec->ops[subIndex].input_tensors_name[0]);
                    subWeightName = "weight_" + subWeightName;
                }

                if (tmpMap == transposeMode || tmpMap == nonTransposeMode) {
                    std::string firstNodeInput =
                        std::string(spec->ops[curIndex].input_tensors_name[0]);
                    std::string lastNodeOutput =
                        std::string(spec->ops[groupLastIndex].output_tensors_name[0]);

                    // reconstruct relationship
                    spec->ops[subIndex].type = OT_LayerNorm;
                    LayerNormParamSpec lnPs;
                    lnPs.axis = -1;
                    lnPs.eps = spec->ops[addIndex].ps.power_spec.shift;
                    spec->ops[subIndex].ps.ln_spec = lnPs;
                    spec->ops[subIndex].num_inputs = 1;
                    mt_free(spec->ops[subIndex].input_tensors_name[1]);
                    str_copy(spec->ops[subIndex].input_tensors_name[0], firstNodeInput.c_str(),
                        NAME_LEN);

                    // TODO new
                    if (tmpMap == transposeMode) {
                        spec->ops[groupLastIndex].type = OT_Transpose;
                        spec->ops[groupLastIndex].ps = spec->ops[transIndex].ps;
                        spec->ops[groupLastIndex].num_inputs = 1;
                        mt_free(spec->ops[groupLastIndex].input_tensors_name[1]);

                        str_copy(spec->ops[groupLastIndex].input_tensors_name[0],
                            spec->ops[subIndex].output_tensors_name[0], NAME_LEN);
                        str_copy(spec->ops[groupLastIndex].output_tensors_name[0],
                            lastNodeOutput.c_str(), NAME_LEN);
                    } else {
                        str_copy(spec->ops[subIndex].output_tensors_name[0], lastNodeOutput.c_str(),
                            NAME_LEN);
                    }

                    // deal with the weight
                    int mulWeightIndex = searchWeightIndex(spec, mulWeightName);
                    int subWeightIndex = searchWeightIndex(spec, subWeightName);
                    float *tmpWeightPtr = (float *)(spec->ws[mulWeightIndex].weight);
                    spec->ws[subWeightIndex].vec =
                        (U8 *)mt_malloc(spec->ws[subWeightIndex].bytes_of_weight);
                    spec->ws[subWeightIndex].bytes_of_vec = spec->ws[subWeightIndex].bytes_of_weight;
                    UNI_MEMCPY(spec->ws[subWeightIndex].vec, spec->ws[subWeightIndex].weight,
                        spec->ws[subWeightIndex].bytes_of_weight);
                    spec->ws[subWeightIndex].bytes_of_weight = 0;
                    mt_free(spec->ws[subWeightIndex].weight);

                    spec->ws[subWeightIndex].weight =
                        (U8 *)mt_malloc(spec->ws[mulWeightIndex].bytes_of_weight);
                    spec->ws[subWeightIndex].bytes_of_weight =
                        spec->ws[mulWeightIndex].bytes_of_weight;
                    UNI_MEMCPY(spec->ws[subWeightIndex].weight, spec->ws[mulWeightIndex].weight,
                        spec->ws[mulWeightIndex].bytes_of_weight);
                    str_copy(spec->ws[subWeightIndex].op_name, spec->ops[subIndex].name, NAME_LEN);
                    spec->ws[mulWeightIndex].bytes_of_weight = 0;
                    mt_free(spec->ws[mulWeightIndex].weight);
                    for (int j = curIndex; j < groupLastIndex; j++) {
                        if (j != subIndex && spec->ops[j].type != OT_None) {
                            setOperatorInvalid(spec, j, false);
                        }
                    }
                    if (tmpMap == nonTransposeMode) {
                        setOperatorInvalid(spec, groupLastIndex, false);
                    }
                    setSharedWeightInvalid(spec, searchOperatorIndexByName(spec, subWeightName));
                    hasOptimized = true;
                }
            }
        }
        return hasOptimized;
    }
};
#endif
