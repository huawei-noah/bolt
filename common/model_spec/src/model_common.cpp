// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "model_common.h"
#include "uni.h"
#include "string_functions.h"

OperatorSpec mt_create_operator(const char *name, OperatorType type, U32 num_inputs, U32 num_outputs)
{
    OperatorSpec newOperator;
    UNI_MEMSET(&(newOperator), 0, sizeof(OperatorSpec));
    U32 length = UNI_MIN(strlen(name), NAME_LEN - 1);
    str_copy(newOperator.name, name, length);
    if (length < NAME_LEN) {
        newOperator.name[length] = '\0';
    }
    newOperator.type = type;
    newOperator.num_inputs = num_inputs;
    newOperator.input_tensors_name = (I8 **)mt_malloc(num_inputs * sizeof(I8 *));
    for (U32 i = 0; i < num_inputs; i++) {
        newOperator.input_tensors_name[i] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
    }
    newOperator.num_outputs = num_outputs;
    newOperator.output_tensors_name = (I8 **)mt_malloc(num_outputs * sizeof(I8 *));
    for (U32 i = 0; i < num_outputs; i++) {
        newOperator.output_tensors_name[i] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
    }
    newOperator.tensor_positions = NULL;
    newOperator.num_quant_feature = 0;
    newOperator.feature_scale = NULL;
    return newOperator;
}

EE mt_insert_operator(ModelSpec *ms, int index, OperatorSpec newOperator)
{
    if (nullptr == ms) {
        return NULL_POINTER;
    }
    OperatorSpec *operatorList =
        (OperatorSpec *)mt_malloc(sizeof(OperatorSpec) * (ms->num_operator_specs + 1));
    for (int i = 0; i < index; i++) {
        operatorList[i] = ms->ops[i];
    }
    operatorList[index] = newOperator;
    for (int i = index; i < ms->num_operator_specs; i++) {
        operatorList[i + 1] = ms->ops[i];
    }
    mt_free(ms->ops);
    ms->ops = operatorList;
    ms->num_operator_specs++;
    return SUCCESS;
}

WeightSpec mt_create_weight(
    const char *name, DataType dataType, U32 bytesOfWeight, U32 bytesOfVec, U32 numQuantScale)
{
    WeightSpec newWeight;
    UNI_MEMSET(&(newWeight), 0, sizeof(WeightSpec));
    U32 length = UNI_MIN(strlen(name), NAME_LEN - 1);
    str_copy(newWeight.op_name, name, length);
    if (length < NAME_LEN) {
        newWeight.op_name[length] = '\0';
    }
    newWeight.mdt = dataType;
    newWeight.bytes_of_weight = bytesOfWeight;
    newWeight.weight = (U8 *)mt_malloc(bytesOfWeight);
    newWeight.bytes_of_vec = bytesOfVec;
    newWeight.vec = (U8 *)mt_malloc(bytesOfVec);
    newWeight.num_quant_scale = numQuantScale;
    newWeight.weight_scale = (QuantSpec *)mt_malloc(sizeof(QuantSpec) * numQuantScale);
    return newWeight;
}

EE mt_insert_weight(ModelSpec *ms, WeightSpec *newWeight, int num)
{
    if (nullptr == ms) {
        return NULL_POINTER;
    }
    WeightSpec *weightList =
        (WeightSpec *)mt_malloc(sizeof(WeightSpec) * (ms->num_weight_specs + num));
    for (int i = 0; i < ms->num_weight_specs; i++) {
        weightList[i] = ms->ws[i];
    }
    for (int i = 0; i < num; ++i) {
        weightList[ms->num_weight_specs + i] = newWeight[i];
    }
    delete ms->ws;
    ms->ws = weightList;
    ms->num_weight_specs += num;
    return SUCCESS;
}

bool isDeprecatedOp(OperatorType opType)
{
    return (opType == OT_None) ? true : false;
}

bool isDeprecatedOpWeight(const ModelSpec *spec, int index)
{
    if (index >= spec->num_weight_specs) {
        return true;
    } else {
        if (spec->ws[index].bytes_of_weight == 0 && spec->ws[index].bytes_of_vec == 0) {
            return true;
        } else {
            return false;
        }
    }
}

void modify_ms_inputs_and_outputs(
    ModelSpec *ms, std::string modifiedInputs, std::string modifiedOutputs)
{
    std::map<std::string, std::string> modifiedStrMap;
    if (modifiedInputs.length() > 0) {
        std::vector<std::string> modified_input_names = split(modifiedInputs, ",");
        if ((I32)(modified_input_names.size()) != ms->num_inputs) {
            UNI_ERROR_LOG("input names not match, please check your params meticulously.\n");
        }
        for (int i = 0; i < ms->num_inputs; i++) {
            std::string tmpStr = modified_input_names[i];
            modifiedStrMap[std::string(ms->input_names[i])] = tmpStr;
            str_copy(ms->input_names[i], tmpStr.c_str(), tmpStr.length());
        }
    }
    if (modifiedOutputs.length() > 0) {
        std::vector<std::string> modified_output_names = split(modifiedOutputs, ",");
        if ((I32)(modified_output_names.size()) != ms->num_outputs) {
            UNI_ERROR_LOG("output names not match, please check your params meticulously.\n");
        }
        for (int i = 0; i < ms->num_outputs; i++) {
            std::string tmpStr = modified_output_names[i];
            modifiedStrMap[std::string(ms->output_names[i])] = tmpStr;
            str_copy(ms->output_names[i], tmpStr.c_str(), tmpStr.length());
        }
    }

    if (modifiedStrMap.size() > 0) {
        for (I32 i = 0; i < ms->num_operator_specs; i++) {
            for (U32 j = 0; j < ms->ops[i].num_inputs; j++) {
                std::string curStr = std::string(ms->ops[i].input_tensors_name[j]);
                if (modifiedStrMap.find(curStr) != modifiedStrMap.end()) {
                    std::string modifiedStr = modifiedStrMap[curStr];
                    str_copy(ms->ops[i].input_tensors_name[j], modifiedStr.c_str(),
                        modifiedStr.length());
                }
            }
            for (U32 j = 0; j < ms->ops[i].num_outputs; j++) {
                std::string curStr = std::string(ms->ops[i].output_tensors_name[j]);
                if (modifiedStrMap.find(curStr) != modifiedStrMap.end()) {
                    std::string modifiedStr = modifiedStrMap[curStr];
                    str_copy(ms->ops[i].output_tensors_name[j], modifiedStr.c_str(),
                        modifiedStr.length());
                }
            }
        }
    }
}
