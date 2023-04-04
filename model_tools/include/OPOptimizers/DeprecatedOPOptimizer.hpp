// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_DEPRECATEDOPOPTIMIZER
#define _H_DEPRECATEDOPOPTIMIZER

#include <map>
#include "OPOptimizer.hpp"

class DeprecatedOPOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;

        for (int i = 0; i < spec->num_weight_specs; i++) {
            weightMap[spec->ws[i].op_name] = i;
        }

        std::map<std::string, int> operatorMap;
        std::set<int> memOutputOP;
        std::set<std::string> outputNames;
        for (int j = 0; j < spec->num_outputs; j++) {
            outputNames.insert(spec->output_names[j]);
        }
        for (int i = 0; i < spec->num_operator_specs; i++) {
            std::string curOut = spec->ops[i].output_tensors_name[0];
            if (outputNames.count(curOut) && spec->ops[i].type != OT_Input &&
                spec->ops[i].type != OT_None && spec->ops[i].type != OT_PreAllocatedMemory) {
                std::string key = this->hash(spec, i);
                memOutputOP.insert(i);  // remember to keep all output ops
                operatorMap[key] = i;   // mark output op key to delete the other redundant ops
            }
        }

        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Pad) {
                if (spec->ops[i].ps.pad_spec.before == 0 && spec->ops[i].ps.pad_spec.after == 0 &&
                    spec->ops[i].ps.pad_spec.top == 0 && spec->ops[i].ps.pad_spec.bottom == 0 &&
                    spec->ops[i].ps.pad_spec.left == 0 && spec->ops[i].ps.pad_spec.right == 0 &&
                    spec->ops[i].ps.pad_spec.front == 0 && spec->ops[i].ps.pad_spec.back == 0) {
                    setOperatorInvalid(spec, i, true);
                    hasOptimized = true;
                    continue;
                }
            }
            if (spec->ops[i].type == OT_Input) {
                spec->ops[i].type = OT_None;
                hasOptimized = true;
                continue;
            }
            if (spec->ops[i].type == OT_Pooling && spec->ops[i].ps.pooling_spec.kernel_t == 1 &&
                spec->ops[i].ps.pooling_spec.kernel_h == 1 &&
                spec->ops[i].ps.pooling_spec.kernel_w == 1 &&
                spec->ops[i].ps.pooling_spec.stride_t == 1 &&
                spec->ops[i].ps.pooling_spec.stride_h == 1 &&
                spec->ops[i].ps.pooling_spec.stride_w == 1 &&
                spec->ops[i].ps.pooling_spec.pad_before == 0 &&
                spec->ops[i].ps.pooling_spec.pad_after == 0 &&
                spec->ops[i].ps.pooling_spec.pad_top == 0 &&
                spec->ops[i].ps.pooling_spec.pad_bottom == 0 &&
                spec->ops[i].ps.pooling_spec.pad_left == 0 &&
                spec->ops[i].ps.pooling_spec.pad_right == 0) {
                if (spec->ops[i].ps.pooling_spec.mode != POOLING_MAX ||
                    (spec->ops[i].ps.pooling_spec.mode == POOLING_MAX &&
                        spec->ops[i].num_outputs == 1)) {
                    setOperatorInvalid(spec, i, true);
                    hasOptimized = true;
                    continue;
                }
                if (spec->ops[i].ps.pooling_spec.mode == POOLING_MAX) {
                    std::vector<std::pair<int, int>> nextOpIndexes = searchOperatorIndexByInput(
                        spec, spec->ops[i].output_tensors_name[0], i + 1, spec->num_operator_specs);
                    if (nextOpIndexes.size() != 0) {
                        continue;
                    }
                    nextOpIndexes = searchOperatorIndexByInput(
                        spec, spec->ops[i].output_tensors_name[1], i + 1, spec->num_operator_specs);
                    if (nextOpIndexes.size() != 1) {
                        continue;
                    }
                    int tfslice = nextOpIndexes[0].first;
                    if (spec->ops[tfslice].type == OT_TfSlice &&
                        spec->ops[tfslice].ps.tfslice_spec.num_dims >= 4 &&
                        spec->ops[tfslice].ps.tfslice_spec.begin[0] == 0 &&
                        spec->ops[tfslice].ps.tfslice_spec.begin[1] == 0 &&
                        spec->ops[tfslice].ps.tfslice_spec.begin[2] == 0 &&
                        spec->ops[tfslice].ps.tfslice_spec.begin[3] == 0 &&
                        spec->ops[tfslice].ps.tfslice_spec.end[0] == -1 &&
                        spec->ops[tfslice].ps.tfslice_spec.end[1] == -1 &&
                        spec->ops[tfslice].ps.tfslice_spec.end[2] == 1 &&
                        spec->ops[tfslice].ps.tfslice_spec.end[3] == 1) {
                        nextOpIndexes = searchOperatorIndexByInput(spec,
                            spec->ops[tfslice].output_tensors_name[0], tfslice + 1,
                            spec->num_operator_specs);
                        if (nextOpIndexes.size() != 1) {
                            continue;
                        }
                        int sub = nextOpIndexes[0].first;
                        if (spec->ops[sub].type == OT_Eltwise &&
                            spec->ops[sub].ps.eltwise_spec.mode == ELTWISE_SUB) {
                            nextOpIndexes = searchOperatorIndexByInput(spec,
                                spec->ops[sub].output_tensors_name[0], sub + 1,
                                spec->num_operator_specs);
                            if (nextOpIndexes.size() != 1) {
                                continue;
                            }
                            int add = nextOpIndexes[0].first;
                            if (spec->ops[add].type == OT_Scale) {
                                int addWeightIndex = searchWeightIndex(spec, spec->ops[add].name);
                                if (addWeightIndex < 0 || spec->ws[addWeightIndex].mdt != DT_I32) {
                                    continue;
                                }
                                int num = spec->ws[addWeightIndex].bytes_of_vec /
                                    bytesOf(spec->ws[addWeightIndex].mdt);
                                int *ptr = (int *)spec->ws[addWeightIndex].vec;
                                if (num < 2) {
                                    continue;
                                }
                                bool merge = true;
                                for (int j = 0; j < num; j++) {
                                    if (ptr[j] / ptr[1] != j) {
                                        merge = false;
                                        break;
                                    }
                                }
                                if (!merge) {
                                    continue;
                                }
                                setOperatorInvalid(spec, i, true);
                                setOperatorInvalid(spec, tfslice, true);
                                setOperatorInvalid(spec, sub, true);
                                setOperatorInvalid(spec, add, true);
                                hasOptimized = true;
                                continue;
                            }
                        }
                    }
                }
            }
            if (spec->ops[i].type == OT_Dropout) {
                setOperatorInvalid(spec, i, true);
                hasOptimized = true;
                continue;
            }
            if (spec->ops[i].type == OT_Expand) {
                bool opt = false;
                for (int j = 0; j < spec->ops[i].ps.expand_spec.num_shape; j++) {
                    opt = true;
                    if (spec->ops[i].ps.expand_spec.shape[j] != 1) {
                        opt = false;
                        break;
                    }
                }
                if (opt) {
                    setOperatorInvalid(spec, i, true);
                    hasOptimized = true;
                }
                continue;
            }
            if (spec->ops[i].type == OT_Slice && spec->ops[i].num_inputs == 1 &&
                spec->ops[i].num_outputs == 1) {
                setOperatorInvalid(spec, i, true);
                hasOptimized = true;
                continue;
            }
            if (spec->ops[i].type == OT_Concat && spec->ops[i].num_inputs == 1) {
                setOperatorInvalid(spec, i, true);
                hasOptimized = true;
                continue;
            }
            if (memOutputOP.count(i)) {
                continue;
            }
            if (spec->ops[i].type != OT_Input && spec->ops[i].type != OT_None &&
                spec->ops[i].type != OT_PreAllocatedMemory) {
                std::string key = this->hash(spec, i);
                if (operatorMap.find(key) == operatorMap.end()) {
                    operatorMap[key] = i;
                } else if (spec->ops[i].num_outputs == 1) {
                    std::string lastOut = spec->ops[operatorMap[key]].output_tensors_name[0];
                    auto nextOpIndexes = searchOperatorIndexByInput(
                        spec, lastOut, operatorMap[key] + 1, spec->num_operator_specs, false);
                    bool opt = true;
                    for (auto nextIdx : nextOpIndexes) {
                        if ((spec->ops[nextOpIndexes[0].first].num_inputs == 1) &&
                            (strcmp(spec->ops[nextOpIndexes[0].first].input_tensors_name[0],
                                 spec->ops[nextOpIndexes[0].first].output_tensors_name[0]) == 0)) {
                            opt = false;
                        }
                    }

                    if (opt) {
                        std::string curOut = spec->ops[i].output_tensors_name[0];
                        nextOpIndexes = searchOperatorIndexByInput(
                            spec, curOut, i + 1, spec->num_operator_specs);

                        setOperatorInvalid(spec, i, false);
                        for (U32 j = 0; j < nextOpIndexes.size(); j++) {
                            str_copy(spec->ops[nextOpIndexes[j].first]
                                         .input_tensors_name[nextOpIndexes[j].second],
                                lastOut.c_str(), NAME_LEN);
                        }
                    }
                }
            }
        }
        for (int i = 0; i < spec->num_inputs; i++) {
            std::string name = spec->input_names[i];
            std::vector<std::pair<int, int>> nextOpIndexes =
                searchOperatorIndexByInput(spec, name, 0, spec->num_operator_specs);
            if (nextOpIndexes.size() == 0) {
                UNI_INFO_LOG("remove model input:%s.\n", name.c_str());
                for (int j = i + 1; j < spec->num_inputs; j++) {
                    spec->input_dims[j - 1] = spec->input_dims[j];
                    str_copy(spec->input_names[j - 1], spec->input_names[j], NAME_LEN);
                }
                mt_free(spec->input_names[spec->num_inputs - 1]);
                spec->num_inputs--;

                int out_idx = -1;
                for (int j = 0; j < spec->num_outputs; j++) {
                    if (spec->output_names[j] == name) {
                        out_idx = j;
                        break;
                    }
                }
                if (out_idx >= 0) {
                    UNI_INFO_LOG("remove model output:%s.\n", name.c_str());
                    for (int j = out_idx + 1; j < spec->num_outputs; j++) {
                        str_copy(spec->output_names[j - 1], spec->output_names[j], NAME_LEN);
                    }
                    mt_free(spec->output_names[spec->num_outputs - 1]);
                    spec->num_outputs--;
                }
                hasOptimized = true;
            }
        }
        return hasOptimized;
    }

    std::string hash(ModelSpec *spec, int i)
    {
        std::string key = std::string(OperatorTypeName()[spec->ops[i].type]);
        for (U32 j = 0; j < spec->ops[i].num_inputs; j++) {
            key += spec->ops[i].input_tensors_name[j] + std::string(",");
        }
        key += " | " +
            copyBuffer<1024>(
                &(spec->ops[i].ps), get_operator_parameter_size(sg_boltVersion, spec->ops[i].type));
        if (weightMap.find(spec->ops[i].name) != weightMap.end()) {
            int weightId = weightMap[spec->ops[i].name];
            if (spec->ws[weightId].bytes_of_weight > 0) {
                int *p = (int *)spec->ws[weightId].weight;
                key += " | " +
                    copyBuffer<1024>(
                        spec->ws[weightId].weight, spec->ws[weightId].bytes_of_weight);
            }
            if (spec->ws[weightId].bytes_of_vec > 0) {
                key += " | " +
                    copyBuffer<1024>(spec->ws[weightId].vec, spec->ws[weightId].bytes_of_vec);
            }
        }
        return key;
    }

private:
    std::map<std::string, int> weightMap;
};
#endif
