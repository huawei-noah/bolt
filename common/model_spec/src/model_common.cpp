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

OperatorSpec mt_create_operator(const char *name, OperatorType type, U32 num_inputs, U32 num_outputs)
{
    OperatorSpec newOperator;
    memset(&(newOperator), 0, sizeof(OperatorSpec));
    U32 length = UNI_MIN(strlen(name), NAME_LEN - 1);
    str_copy(newOperator.name, name, length);
    if (length < NAME_LEN) {
        newOperator.name[length] = '\0';
    }
    newOperator.type = type;
    newOperator.num_inputs = num_inputs;
    newOperator.input_tensors_name = (I8 **)mt_new_storage(num_inputs * sizeof(I8 *));
    for (U32 i = 0; i < num_inputs; i++) {
        newOperator.input_tensors_name[i] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
    }
    newOperator.num_outputs = num_outputs;
    newOperator.output_tensors_name = (I8 **)mt_new_storage(num_outputs * sizeof(I8 *));
    for (U32 i = 0; i < num_outputs; i++) {
        newOperator.output_tensors_name[i] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
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
        (OperatorSpec *)mt_new_storage(sizeof(OperatorSpec) * (ms->num_operator_specs + 1));
    for (int i = 0; i < index; i++) {
        operatorList[i] = ms->ops[i];
    }
    operatorList[index] = newOperator;
    for (int i = index; i < ms->num_operator_specs; i++) {
        operatorList[i + 1] = ms->ops[i];
    }
    delete ms->ops;
    ms->ops = operatorList;
    ms->num_operator_specs++;
    return SUCCESS;
}

WeightSpec mt_create_weight(
    const char *name, DataType dataType, U32 bytesOfWeight, U32 bytesOfVec, U32 numQuantScale)
{
    WeightSpec newWeight;
    memset(&(newWeight), 0, sizeof(WeightSpec));
    U32 length = UNI_MIN(strlen(name), NAME_LEN - 1);
    str_copy(newWeight.op_name, name, length);
    if (length < NAME_LEN) {
        newWeight.op_name[length] = '\0';
    }
    newWeight.mdt = dataType;
    newWeight.bytes_of_weight = bytesOfWeight;
    newWeight.weight = (U8 *)mt_new_storage(bytesOfWeight);
    newWeight.bytes_of_vec = bytesOfVec;
    newWeight.vec = (U8 *)mt_new_storage(bytesOfVec);
    newWeight.num_quant_scale = numQuantScale;
    newWeight.weight_scale = (QuantSpec *)mt_new_storage(sizeof(QuantSpec) * numQuantScale);
    return newWeight;
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

EE str_copy(I8 *dst, const I8 *src, I32 srcLen, I32 dstLen)
{
    memset(dst, 0, dstLen);
    I32 copyLen = NAME_LEN - 1;
    if (copyLen > srcLen) {
        copyLen = srcLen;
    }
    memcpy(dst, src, copyLen * sizeof(I8));
    return SUCCESS;
}

void *mt_new_storage(size_t size)
{
    void *ret = nullptr;
    if (size > 0) {
        try {
            ret = operator new(size);
        } catch (const std::bad_alloc &e) {
            UNI_ERROR_LOG("%s alloc %d bytes failed\n", __FUNCTION__, (int)size);
        }
    }
    return ret;
}

std::string concat_dir_file(std::string dir, std::string file)
{
    std::string ret;
    if (!dir.empty()) {
        int len = dir.size();
        char &last = dir.at(len - 1);
        if ('/' != last) {
            ret = dir + '/';
        } else {
            ret = dir;
        }
        ret += file;
    } else {
        ret = file;
    }

    return ret;
}
