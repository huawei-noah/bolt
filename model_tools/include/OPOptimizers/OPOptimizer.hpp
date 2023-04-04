// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_OPOPTIMIZER
#define _H_OPOPTIMIZER

#include <vector>
#include <set>
#include <string>
#include "model_common.h"
#include "uni.h"
#include "string_functions.h"

static U32 nameIndex = 0;

class OPOptimizer {
public:
    virtual ~OPOptimizer()
    {}

    virtual bool optimize(ModelSpec *spec) = 0;

    template <int SIZE>
    std::string copyBuffer(void *data, int size)
    {
        int length = UNI_MIN(size, SIZE);
        std::string key;
        key.reserve(8 * length);
        U8 *p = (U8 *)data;
        for (int j = 0, i = 0; j < size && i < SIZE; j++) {
            if (p[j] != 0) {
                key += std::to_string(j) + "/" + std::to_string(p[j]) + " ";
                i++;
            }
        }
        return key;
    }

    bool isModelOutput(ModelSpec *spec, int opIndex)
    {
        std::set<std::string> modelOutput;
        for (int i = 0; i < spec->num_outputs; i++) {
            modelOutput.insert(spec->output_names[i]);
        }
        bool ret = false;
        for (unsigned int i = 0; i < spec->ops[opIndex].num_outputs; i++) {
            if (modelOutput.find(spec->ops[opIndex].output_tensors_name[i]) != modelOutput.end()) {
                ret = true;
                break;
            }
        }
        return ret;
    }

    int searchWeightIndex(ModelSpec *spec, std::string op_name)
    {
        if (spec->num_weight_specs <= 0) {
            return -1;
        }

        for (int i = 0; i < spec->num_weight_specs; i++) {
            std::string key = spec->ws[i].op_name;
            if (key == op_name) {
                return i;
            }
        }
        return -1;
    }

    bool isValidOperator(ModelSpec *spec, int index)
    {
        if (index >= spec->num_operator_specs) {
            return false;
        }

        if (spec->ops[index].type != OT_None) {
            return true;
        } else {
            return false;
        }
    }

    void setOperatorInvalid(
        ModelSpec *spec, int index, bool removeEdge = false, int searchStart = -1)
    {
        if (index >= spec->num_operator_specs || index < 0) {
            return;
        }
        UNI_DEBUG_LOG("remove operator:%s(%s) and edges(%d).\n", spec->ops[index].name,
            OperatorTypeName()[spec->ops[index].type], removeEdge);
        spec->ops[index].type = OT_None;
        int weightId = searchWeightIndex(spec, spec->ops[index].name);
        if (weightId >= 0) {
            setWeightOperatorInvalid(spec, weightId);
        }
        if (removeEdge) {
            if (spec->ops[index].num_inputs == 1 && spec->ops[index].num_outputs == 1 &&
                std::string(spec->ops[index].input_tensors_name[0]) ==
                    std::string(spec->ops[index].output_tensors_name[0])) {
                return;
            }
            if (spec->ops[index].num_inputs > 0) {
                for (U32 i = 0; i < spec->ops[index].num_outputs; i++) {
                    if (searchStart == -1) {
                        searchStart = index + 1;
                    }
                    std::vector<std::pair<int, int>> operatorIndexes0 = searchOperatorIndexByInput(
                        spec, spec->ops[index].output_tensors_name[i], searchStart);
                    for (U32 j = 0; j < operatorIndexes0.size(); j++) {
                        str_copy(spec->ops[operatorIndexes0[j].first]
                                     .input_tensors_name[operatorIndexes0[j].second],
                            spec->ops[index].input_tensors_name[0], NAME_LEN);
                    }
                    std::vector<int> outputs = searchString(spec->output_names, spec->num_outputs,
                        spec->ops[index].output_tensors_name[i]);
                    for (U32 j = 0; j < outputs.size(); j++) {
                        str_copy(spec->output_names[outputs[j]],
                            spec->ops[index].input_tensors_name[0], NAME_LEN);
                    }
                }
            }
        }
    }

    void setWeightOperatorInvalid(ModelSpec *spec, int index)
    {
        if (index >= spec->num_weight_specs || index < 0) {
            return;
        }
        UNI_DEBUG_LOG("remove weight operator:%s.\n", spec->ws[index].op_name);
        spec->ws[index].bytes_of_weight = 0;
        mt_free(spec->ws[index].weight, spec);
        spec->ws[index].bytes_of_vec = 0;
        mt_free(spec->ws[index].vec, spec);
        spec->ws[index].num_quant_scale = 0;
        mt_free(spec->ws[index].weight_scale, spec);
    }

    void setSharedWeightInvalid(ModelSpec *spec, int index)
    {
        if (index >= spec->num_operator_specs || index < 0 ||
            spec->ops[index].type != OT_SharedWeight) {
            return;
        }
        auto next = searchOperatorIndexByInput(spec, spec->ops[index].output_tensors_name[0], index);
        if (next.size() > 0) {
            return;
        }
        setOperatorInvalid(spec, index);
    }

    int searchOperatorIndexByName(ModelSpec *spec, std::string name)
    {
        int result = -1;
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].name == name) {
                result = i;
                break;
            }
        }
        return result;
    }

    int searchOperatorIndexBackward(ModelSpec *spec,
        int end,
        OperatorType *queryOps,
        int queryNum,
        bool unskip = true,
        int start = 0)
    {
        for (int i = end; i >= start; i--) {
            if (isValidOperator(spec, i)) {
                for (int j = 0; j < queryNum; j++) {
                    OperatorType opType = queryOps[j];
                    if (spec->ops[i].type == opType) {
                        return i;
                    }
                }
                if (unskip) {
                    return -1;
                }
            }
        }
        return -1;
    }

    int searchOperatorIndexForward(ModelSpec *spec,
        int start,
        OperatorType *queryOps,
        int queryNum,
        bool unskip = true,
        std::string str = "",
        int end = 0)
    {
        std::string strEmpty = "";
        if (end == 0) {
            end = spec->num_operator_specs;
        }
        for (int i = start; i < end; i++) {
            if (isValidOperator(spec, i)) {
                for (int j = 0; j < queryNum; j++) {
                    OperatorType opType = queryOps[j];
                    if (spec->ops[i].type == opType) {
                        if (str != strEmpty && spec->ops[i].num_inputs > 0) {
                            std::string inputName0 =
                                spec->ops[i].input_tensors_name[0];  // May examine more inputs in the future
                            if (inputName0 != str) {
                                continue;
                            }
                        }
                        return i;
                    }
                }
                if (unskip) {
                    return -1;
                }
            }
        }
        return -1;
    }

    std::vector<std::pair<int, int>> searchOperatorIndexByInput(ModelSpec *spec,
        std::string tensorName,
        int left = 0,
        int right = 0,
        bool nearestNeighbor = true)
    {
        std::vector<std::pair<int, int>> result;
        if (right == 0) {
            right = spec->num_operator_specs;
        }
        bool hasFind = false;
        for (int i = left; i < right; i++) {
            if (isValidOperator(spec, i)) {
                for (int j = 0; j < (int)spec->ops[i].num_inputs; j++) {
                    if (spec->ops[i].input_tensors_name[j] == tensorName) {
                        result.push_back(std::make_pair(i, j));
                        hasFind = true;
                    }
                }
                if (nearestNeighbor) {
                    for (int j = 0; j < (int)spec->ops[i].num_outputs; j++) {
                        if (spec->ops[i].output_tensors_name[j] == tensorName && hasFind) {
                            return result;
                        }
                    }
                }
            }
        }
        return result;
    }

    std::vector<std::pair<int, int>> searchOperatorIndexByOutput(ModelSpec *spec,
        std::string tensorName,
        int left = 0,
        int right = 0,
        bool nearestNeighbor = true)
    {
        std::vector<std::pair<int, int>> result;
        if (right == 0) {
            right = spec->num_operator_specs;
        }
        bool hasFind = false;
        for (int i = right - 1; i >= left; i--) {
            if (isValidOperator(spec, i)) {
                for (int j = 0; j < (int)spec->ops[i].num_outputs; j++) {
                    if (spec->ops[i].output_tensors_name[j] == tensorName) {
                        result.push_back(std::make_pair(i, j));
                        hasFind = true;
                    }
                }
                if (nearestNeighbor) {
                    for (int j = 0; j < (int)spec->ops[i].num_inputs; j++) {
                        if (spec->ops[i].input_tensors_name[j] == tensorName && hasFind) {
                            return result;
                        }
                    }
                }
            }
        }
        return result;
    }

    std::vector<int> searchString(char **array, int num, const char *data)
    {
        std::vector<int> result;
        for (int i = 0; i < num; i++) {
            if (std::string(array[i]) == std::string(data)) {
                result.push_back(i);
            }
        }
        return result;
    }

    int skipInvalidOperator(ModelSpec *spec, int k)
    {
        for (; k < spec->num_operator_specs; k++) {
            if (isValidOperator(spec, k)) {
                break;
            }
        }
        return k;
    }

    std::string allocName(std::string suggest = "")
    {
        std::string name = suggest;
        if (name == "" || name.length() >= NAME_LEN) {
            name = "_" + int2Any(nameIndex, 60);
            nameIndex += 1;
        }
        if (name.length() >= NAME_LEN) {
            UNI_ERROR_LOG("NAME_LEN(%d) is not big enough to store %s.\n", NAME_LEN, name.c_str());
        }
        return name;
    }

};
#endif
