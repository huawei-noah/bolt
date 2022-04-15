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

EE serialize_header(const ModelSpec *spec, std::string *tmp)
{
    U32 bufSize = sizeof(I32) * 2 + sizeof(I8) * NAME_LEN + sizeof(DataType) + sizeof(I32) +
        sizeof(I8) * NAME_LEN * spec->num_inputs + sizeof(TensorDesc) * spec->num_inputs +
        sizeof(I32) + sizeof(I8) * NAME_LEN * spec->num_outputs;
    I8 *data = (I8 *)mt_new_storage(bufSize);

    I32 *pointer4version = (I32 *)data;
    memcpy(pointer4version, &spec->version, sizeof(I32));
    pointer4version += 1;

    I32 *pointer4magicNumber = (I32 *)pointer4version;
    memcpy(pointer4magicNumber, &spec->magic_number, sizeof(I32));
    pointer4magicNumber += 1;

    I8 *pointer4modelName = (I8 *)pointer4magicNumber;
    str_copy(pointer4modelName, spec->model_name, NAME_LEN);
    pointer4modelName += NAME_LEN;

    DataType *pointer4dt = (DataType *)pointer4modelName;
    *pointer4dt = spec->dt;
    pointer4dt++;

    I32 *pointer4numInputs = (I32 *)pointer4dt;
    *pointer4numInputs = spec->num_inputs;
    pointer4numInputs++;

    I8 *pointer4InputNames = (I8 *)pointer4numInputs;
    for (int i = 0; i < spec->num_inputs; i++) {
        str_copy(pointer4InputNames, spec->input_names[i], NAME_LEN);
        pointer4InputNames += NAME_LEN;
    }

    TensorDesc *pointer4TensorDesc = (TensorDesc *)pointer4InputNames;
    memcpy(pointer4TensorDesc, spec->input_dims, sizeof(TensorDesc) * spec->num_inputs);
    pointer4TensorDesc += spec->num_inputs;

    I32 *pointer4numOutputs = (I32 *)pointer4TensorDesc;
    *pointer4numOutputs = spec->num_outputs;
    pointer4numOutputs++;

    I8 *pointer4outputNames = (I8 *)pointer4numOutputs;
    for (int i = 0; i < spec->num_outputs; i++) {
        str_copy(pointer4outputNames, spec->output_names[i], NAME_LEN);
        pointer4outputNames += NAME_LEN;
    }

    tmp->clear();
    CHECK_REQUIREMENT((U32)(pointer4outputNames - data) == bufSize);
    tmp->assign(data, data + bufSize);
    delete data;
    return SUCCESS;
}

U32 operator_memory_size(OperatorSpec *ops)
{
    // sizeof(U32) * 4 : type + num_inputs + num_output + num_quant_feature
    U32 allocatedBufferSize = sizeof(I8) * NAME_LEN + sizeof(U32) * 4 +
        ops->num_inputs * NAME_LEN * sizeof(I8) + ops->num_outputs * NAME_LEN * sizeof(I8) +
        (ops->num_inputs + ops->num_outputs) * sizeof(I32) + get_operator_parameter_size(ops->type);

    for (U32 i = 0; i < ops->num_quant_feature; i++) {
        allocatedBufferSize += sizeof(int);  // num_scale
        allocatedBufferSize += ops->feature_scale[i].num_scale * sizeof(F32);
    }
    return allocatedBufferSize;
}

EE serialize_operators(const ModelSpec *spec, std::string *tmp)
{
    OperatorSpec *opsTmp = spec->ops;
    int removeOpNum = 0;
    U32 bufSize = sizeof(I32);
    for (int i = 0; i < spec->num_operator_specs; i++) {
        if (isDeprecatedOp(opsTmp->type)) {
            removeOpNum++;
        } else {
            bufSize += operator_memory_size(opsTmp);
        }
        opsTmp++;
    }

    char *data = (char *)mt_new_storage(bufSize);

    I32 *pointer4numOperatorSpecs = (I32 *)data;
    *pointer4numOperatorSpecs = spec->num_operator_specs - removeOpNum;  // attention
    pointer4numOperatorSpecs++;

    OperatorSpec *opsPointer = spec->ops;
    I8 *pointer4opsName = (I8 *)pointer4numOperatorSpecs;

    for (int i = 0; i < spec->num_operator_specs; i++) {
        if (isDeprecatedOp(opsPointer[i].type)) {
            continue;
        }

        str_copy(pointer4opsName, opsPointer[i].name, NAME_LEN);  // to copy the name of op
        pointer4opsName += NAME_LEN;

        U32 *pointer4opsType = (U32 *)pointer4opsName;
        *pointer4opsType = opsPointer[i].type;
        pointer4opsType++;

        U32 *pointer4opsNumInputs = pointer4opsType;
        *pointer4opsNumInputs = opsPointer[i].num_inputs;
        pointer4opsNumInputs++;

        I8 *pointer4opsInputTensorsName = (I8 *)pointer4opsNumInputs;
        for (U32 j = 0; j < opsPointer[i].num_inputs; j++) {
            str_copy(pointer4opsInputTensorsName, opsPointer[i].input_tensors_name[j], NAME_LEN);
            pointer4opsInputTensorsName += NAME_LEN;
        }

        U32 *pointer4opsNumOutputs = (U32 *)pointer4opsInputTensorsName;
        *pointer4opsNumOutputs = opsPointer[i].num_outputs;
        pointer4opsNumOutputs++;

        I8 *pointer4opsOutputTensorsName = (I8 *)pointer4opsNumOutputs;
        for (U32 j = 0; j < opsPointer[i].num_outputs; j++) {
            str_copy(pointer4opsOutputTensorsName, opsPointer[i].output_tensors_name[j], NAME_LEN);
            pointer4opsOutputTensorsName += NAME_LEN;
        }

        I32 *pointer4tensorPos = (I32 *)pointer4opsOutputTensorsName;
        U32 numTensors = opsPointer[i].num_inputs + opsPointer[i].num_outputs;
        if (nullptr != opsPointer[i].tensor_positions) {
            memcpy(pointer4tensorPos, opsPointer[i].tensor_positions, numTensors * sizeof(I32));
        } else {
            for (U32 j = 0; j < numTensors; j++) {
                pointer4tensorPos[j] = -1;
            }
        }
        pointer4tensorPos += numTensors;

        U32 *pointer4numint8 = (U32 *)pointer4tensorPos;
        *pointer4numint8 = opsPointer[i].num_quant_feature;
        pointer4numint8++;

        int *pointer4quant = (int *)pointer4numint8;
        for (U32 j = 0; j < opsPointer[i].num_quant_feature; j++) {
            *pointer4quant = opsPointer[i].feature_scale[j].num_scale;
            int num = *pointer4quant;
            pointer4quant++;
            memcpy(pointer4quant, opsPointer[i].feature_scale[j].scale, num * sizeof(F32));
            pointer4quant += num;
        }

        char *pointer4parameterSpecs = (char *)pointer4quant;
        int operatorParameterSize = get_operator_parameter_size(opsPointer[i].type);
        memcpy(pointer4parameterSpecs, &(opsPointer[i].ps), operatorParameterSize);
        pointer4parameterSpecs += operatorParameterSize;
        pointer4opsName = (I8 *)pointer4parameterSpecs;
    }

    tmp->clear();
    CHECK_REQUIREMENT((U32)(pointer4opsName - data) == bufSize);
    tmp->assign(data, data + bufSize);
    delete data;
    return SUCCESS;
}

EE serialize_weights(const ModelSpec *spec, std::string *tmp)
{
    WeightSpec *tmpPointer = spec->ws;
    U32 bufSize = sizeof(I32);
    U32 weightCount = 0;
    for (int i = 0; i < spec->num_weight_specs; i++) {
        if (isDeprecatedOpWeight(spec, i)) {
            continue;
        }

        // U32 x 5: length, mdt, bytes_of_weight, bytes_of_vec, num_quant_scale
        bufSize += sizeof(I8) * NAME_LEN + sizeof(U32) * 5 + tmpPointer[i].bytes_of_weight +
            tmpPointer[i].bytes_of_vec;
        for (U32 j = 0; j < tmpPointer[i].num_quant_scale; j++) {
            bufSize += sizeof(int);  // num_scale
            bufSize += tmpPointer[i].weight_scale[j].num_scale * sizeof(F32);
        }

        weightCount++;
    }
    char *data = (char *)mt_new_storage(bufSize);

    I32 *pointer4numWeightSpecs = (I32 *)data;
    *pointer4numWeightSpecs = weightCount;
    pointer4numWeightSpecs++;

    WeightSpec *wsPointer = spec->ws;
    char *pointer4wsOpName = (char *)pointer4numWeightSpecs;
    for (int i = 0; i < spec->num_weight_specs; i++) {
        if (isDeprecatedOpWeight(spec, i)) {
            continue;
        }

        U32 *length = (U32 *)pointer4wsOpName;
        U32 len;
        len = wsPointer[i].bytes_of_weight + wsPointer[i].bytes_of_vec;
        *length = len;
        pointer4wsOpName += sizeof(U32);

        str_copy(pointer4wsOpName, wsPointer[i].op_name, NAME_LEN);
        pointer4wsOpName += NAME_LEN;

        U32 *pointer4wsMdt = (U32 *)pointer4wsOpName;
        *pointer4wsMdt = wsPointer[i].mdt;
        pointer4wsMdt++;

        U32 *pointer4wsBytesOfWeight = (U32 *)pointer4wsMdt;
        *pointer4wsBytesOfWeight = wsPointer[i].bytes_of_weight;
        pointer4wsBytesOfWeight++;

        U8 *pointer4wsWeight = (U8 *)pointer4wsBytesOfWeight;
        memcpy(pointer4wsWeight, wsPointer[i].weight, wsPointer[i].bytes_of_weight);
        pointer4wsWeight += wsPointer[i].bytes_of_weight;

        U32 *pointer4wsBytesOfVec = (U32 *)pointer4wsWeight;
        *pointer4wsBytesOfVec = wsPointer[i].bytes_of_vec;
        pointer4wsBytesOfVec++;

        U8 *pointer4wsVec = (U8 *)pointer4wsBytesOfVec;
        memcpy(pointer4wsVec, wsPointer[i].vec, wsPointer[i].bytes_of_vec);
        pointer4wsVec += wsPointer[i].bytes_of_vec;

        U32 *pointer4numquant = (U32 *)pointer4wsVec;
        *pointer4numquant = wsPointer[i].num_quant_scale;
        pointer4numquant++;

        int *pointer4quant = (int *)pointer4numquant;
        for (U32 j = 0; j < wsPointer[i].num_quant_scale; j++) {
            *pointer4quant = wsPointer[i].weight_scale[j].num_scale;
            int num = *pointer4quant;
            pointer4quant++;
            memcpy(pointer4quant, wsPointer[i].weight_scale[j].scale, num * sizeof(F32));
            pointer4quant += num;
        }

        pointer4wsOpName = (char *)pointer4quant;
    }

    tmp->clear();
    CHECK_REQUIREMENT((U32)(pointer4wsOpName - data) == bufSize);
    tmp->assign(data, data + bufSize);
    delete data;
    return SUCCESS;
}

EE serialize_model(const ModelSpec *spec, std::string *bytes)
{
    bytes->clear();
    std::string tmp;

    CHECK_STATUS(serialize_header(spec, &tmp));
    *bytes += tmp;

    CHECK_STATUS(serialize_operators(spec, &tmp));
    *bytes += tmp;

    CHECK_STATUS(serialize_weights(spec, &tmp));
    *bytes += tmp;
    return SUCCESS;
}

EE write_to_file(std::string *bytes, const char *fn)
{
    FILE *file = fopen(fn, "wb");
    if (file == NULL) {
        UNI_ERROR_LOG("Cannot write bolt model to %s.\n", fn);
        return FILE_ERROR;
    }
    U32 size = fwrite(bytes->c_str(), sizeof(char), bytes->size(), file);
    if (size != bytes->size()) {
        UNI_ERROR_LOG("Write bolt model file %s failed.\n", fn);
        return FILE_ERROR;
    }
    int status = fclose(file);
    if (status != 0) {
        UNI_ERROR_LOG("Close bolt model file %s write handle failed.\n", fn);
        return FILE_ERROR;
    }
    return SUCCESS;
}

EE serialize_model_to_file(const ModelSpec *spec, const char *fn)
{
    UNI_DEBUG_LOG("Write bolt model to %s...\n", fn);
    std::string bytes = "";
    CHECK_STATUS(serialize_model(spec, &bytes));
    CHECK_STATUS(write_to_file(&bytes, fn));
    UNI_DEBUG_LOG("Write bolt model end.\n");
    return SUCCESS;
}
