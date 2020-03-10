// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <string>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include "model_serialize_deserialize.hpp"
#include "model_tools.h"
#include "model_optimizer.hpp"
#include "OPOptimizers/DeprecatedOPOptimizer.hpp"

EE opeator_relationship(ModelSpec* spec) {
    std::map<std::string, bool> opCanInChange;
    std::set<std::string> inplaceTensors;
    std::map<std::string, int> inplaceTensorInNum;
    std::map<std::string, int> inplaceTensorOutNum;
    std::map<std::string, std::vector<std::string>> opInTensorNew;
    std::map<std::string, std::string> opOutTensorNew;
    std::map<std::string, std::string> tensorOpMapping;
    std::map<std::string, std::vector<std::string>> tensorFlowsToOpSet;

    for (int i = 0; i < spec->num_operator_specs; i++) {
        if (spec->ops[i].num_inputs == 1 && spec->ops[i].num_outputs == 1) {
            std::string inputTensorStr = spec->ops[i].input_tensors_name[0];
            std::string outputTensorStr = spec->ops[i].output_tensors_name[0];
            if (inputTensorStr.compare(outputTensorStr) == 0) {
                inplaceTensors.insert(inputTensorStr);
                opCanInChange.insert(std::make_pair(inputTensorStr, true));
            }
        }
    }

    for (int i = 0; i < spec->num_operator_specs; i++) {
        std::string currentOpName = spec->ops[i].name;
        int in_tensor_number = spec->ops[i].num_inputs;
        std::vector<std::string> inTensorVec;

        // dealing with the relationship of  op -- input tensors
        for (int j = 0; j < in_tensor_number; j++) {
            std::string tmpInTensor = spec->ops[i].input_tensors_name[j];
            if (inplaceTensors.find(tmpInTensor) != inplaceTensors.end()) {  // juduge inplace op or not
                int inId;
                if (inplaceTensorInNum.find(tmpInTensor) == inplaceTensorInNum.end()) {
                    inId = 1;
                    inplaceTensorInNum.insert(std::make_pair(tmpInTensor, inId));
                    opCanInChange[tmpInTensor] = true;
                }else{
                    if (opCanInChange[tmpInTensor] == false) {
                        inId = inplaceTensorInNum[tmpInTensor]+1;
                        // inplaceTensorInNum.insert(std::make_pair(tmpInTensor, inId));
                        inplaceTensorInNum[tmpInTensor] = inId;
                        opCanInChange[tmpInTensor] = true;
                    }else{
                        inId = inplaceTensorInNum[tmpInTensor];
                        opCanInChange[tmpInTensor] = true;
                    }
                }
                std::ostringstream stream;
                stream << inId;
                std::string tmpInTensorChanged = tmpInTensor + "_" + stream.str();
                inTensorVec.push_back(tmpInTensorChanged);

                if (tensorFlowsToOpSet.find(tmpInTensorChanged) == tensorFlowsToOpSet.end()) {
                    std::vector<std::string> tmpVector;
                    tmpVector.push_back(currentOpName);
                    tensorFlowsToOpSet.insert(std::make_pair(tmpInTensorChanged, tmpVector));
                }else{
                    tensorFlowsToOpSet[tmpInTensorChanged].push_back(currentOpName);
                }

            }else{
                inTensorVec.push_back(tmpInTensor);

                if (tensorFlowsToOpSet.find(tmpInTensor) == tensorFlowsToOpSet.end()) {
                    std::vector<std::string> tmpVector;
                    tmpVector.push_back(currentOpName);
                    tensorFlowsToOpSet.insert(std::make_pair(tmpInTensor, tmpVector));
                }else{
                    tensorFlowsToOpSet[tmpInTensor].push_back(currentOpName);
                }
            }
        }
        opInTensorNew.insert(std::make_pair(currentOpName, inTensorVec));

        // dealing with the relationship of op -- output tensors 
        std::string tmpOutTensor = spec->ops[i].output_tensors_name[0];
        if (inplaceTensors.find(tmpOutTensor) != inplaceTensors.end()) {
            // todo
            int outId;
            if (inplaceTensorOutNum.find(tmpOutTensor) == inplaceTensorOutNum.end()) {
                outId = 1;
                inplaceTensorOutNum.insert(std::make_pair(tmpOutTensor, outId));
                opCanInChange[tmpOutTensor] = false;
            }else{
                outId = inplaceTensorOutNum[tmpOutTensor] + 1;
                // inplaceTensorOutNum.insert(std::make_pair(tmpOutTensor, outId)); can not update
                inplaceTensorOutNum[tmpOutTensor] = outId;
                opCanInChange[tmpOutTensor] = false;
            }
            std::ostringstream stream;
            stream << outId;
            std::string tmpOutTensorChanged = tmpOutTensor + "_" + stream.str();
            opOutTensorNew.insert(std::make_pair(currentOpName, tmpOutTensorChanged));
            tensorOpMapping.insert(std::make_pair(tmpOutTensorChanged, currentOpName));
        }else{
            opOutTensorNew.insert(std::make_pair(currentOpName, tmpOutTensor));
            tensorOpMapping.insert(std::make_pair(tmpOutTensor, currentOpName));
        }
    }

    // assign op-op relationship
    int opNum = spec->num_operator_specs;
    spec->num_op_tensor_entries = opNum;
    OperatorSpec* opsPtr2 = spec->ops;
    OperatorRelationshipMapEntry* oprmePtr = (OperatorRelationshipMapEntry*)mt_new_storage(sizeof(OperatorRelationshipMapEntry) * opNum);
    spec->op_relationship_entries = oprmePtr;
    for (int j = 0; j < opNum; j++) {
        str_copy(oprmePtr[j].op,  opsPtr2[j].name, NAME_LEN);
        int opInOpNum = opInTensorNew[opsPtr2[j].name].size();
        oprmePtr[j].num_inputs = opInOpNum;
        oprmePtr[j].input_op_names = (I8 **)mt_new_storage(opInOpNum * sizeof(I8 *));
        for (int k = 0; k < opInOpNum; k++) {
            oprmePtr[j].input_op_names[k] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
            std::string ten_name = opInTensorNew[opsPtr2[j].name][k];
            std::string tensor2op = tensorOpMapping[ten_name];
            str_copy(oprmePtr[j].input_op_names[k], tensor2op.c_str(), tensor2op.length());
        }

        int opOutOpNum = tensorFlowsToOpSet[opOutTensorNew[opsPtr2[j].name]].size();
        oprmePtr[j].num_outputs = opOutOpNum;
        oprmePtr[j].output_op_names = (I8 **)mt_new_storage(opOutOpNum * sizeof(I8 *));
        for (int k = 0; k < opOutOpNum; k++) {
            oprmePtr[j].output_op_names[k] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
            std::string tensor2op = tensorFlowsToOpSet[opOutTensorNew[opsPtr2[j].name]][k];
            str_copy(oprmePtr[j].output_op_names[k], tensor2op.c_str(), tensor2op.length());
        }
    }
    return SUCCESS;
}


EE serialize_header(const ModelSpec* spec, std::string* tmp) {
    U32 bufSize = sizeof(I32) * 2  \
                 + sizeof(I8) * NAME_LEN + sizeof(DataType) + sizeof(I32) \
                 + sizeof(I8) * NAME_LEN * spec->num_inputs + sizeof(TensorDesc) * spec->num_inputs \
                 + sizeof(I32) + sizeof(I8) * NAME_LEN * spec->num_outputs;
    I8* data = (I8*)mt_new_storage(bufSize);

    I32* pointer4version = (I32*)data;
    memcpy(pointer4version, &spec->version, sizeof(I32));
    pointer4version += 1;   // the pointer datatype(I32) of add 1 means 4 steps

    I32* pointer4magicNumber = (I32*)pointer4version;
    memcpy(pointer4magicNumber, &spec->magic_number, sizeof(I32));
    pointer4magicNumber += 1;

    I8* pointer4modelName = (I8*)pointer4magicNumber;
    str_copy(pointer4modelName, spec->model_name, NAME_LEN);
    pointer4modelName += NAME_LEN;

    DataType* pointer4dt = (DataType*)pointer4modelName;
    *pointer4dt = spec->dt;
    pointer4dt++;

    I32* pointer4numInputs = (I32*)pointer4dt;
    *pointer4numInputs = spec->num_inputs;
    pointer4numInputs++;

    I8* pointer4InputNames = (I8*)pointer4numInputs;
    for (int i = 0; i < spec->num_inputs; i++) {
        str_copy(pointer4InputNames, spec->input_names[i], NAME_LEN);
        pointer4InputNames += NAME_LEN;
    }

    TensorDesc* pointer4TensorDesc = (TensorDesc*)pointer4InputNames;
    memcpy(pointer4TensorDesc, spec->input_dims, sizeof(TensorDesc) * spec->num_inputs);
    pointer4TensorDesc += spec->num_inputs;

    I32* pointer4numOutputs = (I32 *)pointer4TensorDesc;
    *pointer4numOutputs = spec->num_outputs;
    pointer4numOutputs++;

    I8* pointer4outputNames = (I8 *)pointer4numOutputs;
    for (int i = 0; i < spec->num_outputs; i++) {
        str_copy(pointer4outputNames, spec->output_names[i], NAME_LEN);
        pointer4outputNames += NAME_LEN;
    }

    tmp->clear();
    CHECK_REQUIREMENT(pointer4outputNames - data == bufSize);
    tmp->assign(data, data + bufSize);
    delete [] data;
    return SUCCESS;
}


EE deserialize_header(const char* bytes, ModelSpec* spec, U32* pos)
{
    const char* pointer = bytes + *pos;
    memcpy(&spec->version, pointer, sizeof(I32));
    pointer += sizeof(I32);
    *pos += sizeof(I32);
    if (spec->version != mt_version()) {
        std::cerr << "[ERROR] version not_match: code " << mt_version() << \
                     "bolt model " << spec->version << std::endl;
        CHECK_STATUS(NOT_MATCH);
    }

    memcpy(&spec->magic_number, pointer, sizeof(I32));
    pointer += sizeof(I32);
    *pos += sizeof(I32);
    if (spec->magic_number != mt_magic_number()) {
        std::cerr << "[ERROR] magic_number not_match: code " << mt_magic_number() << \
                     "bolt model " << spec->version << std::endl;
        CHECK_STATUS(NOT_MATCH);
    }

    str_copy(spec->model_name, pointer, NAME_LEN);
    pointer += NAME_LEN;
    *pos += NAME_LEN;

    spec->dt = *((DataType*)pointer);
    pointer += sizeof(DataType);
    *pos += sizeof(DataType);

    spec->num_inputs = *((I32*)pointer);
    pointer += sizeof(I32);
    *pos += sizeof(I32);

    spec->input_names = (I8**)mt_new_storage(spec->num_inputs * sizeof(I8*));
    for (int i = 0; i < spec->num_inputs; i++) {
        spec->input_names[i] = (I8*)mt_new_storage(NAME_LEN * sizeof(I8));
        str_copy(spec->input_names[i], pointer, NAME_LEN);
        pointer += NAME_LEN;
        *pos += NAME_LEN;
    }

    spec->input_dims = (TensorDesc *)mt_new_storage(spec->num_inputs * sizeof(TensorDesc));
    memcpy(spec->input_dims, pointer, spec->num_inputs * sizeof(TensorDesc));
    pointer += spec->num_inputs * sizeof(TensorDesc);
    *pos += spec->num_inputs * sizeof(TensorDesc);

    spec->num_outputs = *((I32*)pointer);
    pointer += sizeof(I32);
    *pos += sizeof(I32);

    spec->output_names = (I8**)mt_new_storage(spec->num_outputs * NAME_LEN);
    for (int i = 0; i < spec->num_outputs; i++) {
        spec->output_names[i] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
        str_copy(spec->output_names[i], pointer, NAME_LEN);
        pointer += NAME_LEN;
        *pos += NAME_LEN;
    }
    return SUCCESS;
}

U32 operator_memory_size(OperatorSpec* ops) {
    // sizeof(U32) * 3 : type + num_inputs + num_output
    U32 allocatedBufferSize = sizeof(I8) * NAME_LEN + sizeof(U32) * 3
                       + ops->num_inputs * NAME_LEN * sizeof(I8)
                       + ops->num_outputs * NAME_LEN * sizeof(I8)
                       + (ops->num_inputs + ops->num_outputs) * sizeof(I32)
                       + sizeof(ParameterSpec);
    switch (ops->type) {
        case OT_Eltwise: {
            allocatedBufferSize += ops->ps.eltwise_spec.elt_sum_spec.coeff_size * sizeof(float);
            break;
        }
        default:
            break;
    }
    return allocatedBufferSize;
}


EE serialize_operators(const ModelSpec* spec, std::string* tmp) {
    OperatorSpec* opsTmp = spec->ops;
    int removeOpNum = 0;
    U32 bufSize = sizeof(I32);
    for (int i = 0; i < spec->num_operator_specs; i++) {
        if (DeprecatedOPOptimizer::isDeprecatedOp(opsTmp->type)) {
            removeOpNum++;
        }
        else {
            bufSize += operator_memory_size(opsTmp);
        }
        opsTmp++;
    }

    char* data = (char*)mt_new_storage(bufSize);

    I32* pointer4numOperatorSpecs = (I32 *)data;
    *pointer4numOperatorSpecs = spec->num_operator_specs - removeOpNum;  // attention
    pointer4numOperatorSpecs++;

    OperatorSpec* opsPointer = spec->ops;
    I8* pointer4opsName = (I8*)pointer4numOperatorSpecs;

    for (int i = 0; i < spec->num_operator_specs; i++) {
        if (DeprecatedOPOptimizer::isDeprecatedOp(opsPointer[i].type)) {
            continue;
        }

        str_copy(pointer4opsName, opsPointer[i].name, NAME_LEN);    // to copy the name of op
        pointer4opsName += NAME_LEN;

        U32* pointer4opsType = (U32 *)pointer4opsName;
        *pointer4opsType = opsPointer[i].type;
        pointer4opsType++;

        U32* pointer4opsNumInputs = pointer4opsType;
        *pointer4opsNumInputs = opsPointer[i].num_inputs;
        pointer4opsNumInputs++;

        I8* pointer4opsInputTensorsName = (I8 *)pointer4opsNumInputs;
        for (U32 j = 0; j < opsPointer[i].num_inputs; j++) {
            str_copy(pointer4opsInputTensorsName, opsPointer[i].input_tensors_name[j], NAME_LEN);
            pointer4opsInputTensorsName += NAME_LEN;
        }

        U32* pointer4opsNumOutputs = (U32 *)pointer4opsInputTensorsName;
        *pointer4opsNumOutputs = opsPointer[i].num_outputs;
        pointer4opsNumOutputs++;

        I8* pointer4opsOutputTensorsName = (I8 *)pointer4opsNumOutputs;
        for (U32 j = 0; j < opsPointer[i].num_outputs; j++) {
            str_copy(pointer4opsOutputTensorsName, opsPointer[i].output_tensors_name[j], NAME_LEN);
            pointer4opsOutputTensorsName += NAME_LEN;
        }

        I32* pointer4tensorPos = (I32*)pointer4opsOutputTensorsName;
        U32 numTensors = opsPointer[i].num_inputs + opsPointer[i].num_outputs;
        if (nullptr != opsPointer[i].tensor_positions) {
            memcpy(pointer4tensorPos, opsPointer[i].tensor_positions, numTensors*sizeof(I32));
        } else {
            memset(pointer4tensorPos, 0, numTensors*sizeof(I32));
        }
        pointer4tensorPos += numTensors;

        char* pointer4parameterSpecs = (char *)pointer4tensorPos;
        memcpy(pointer4parameterSpecs, &(opsPointer[i].ps), sizeof(ParameterSpec));
        pointer4parameterSpecs += sizeof(ParameterSpec);
        switch (opsPointer[i].type) {
            case OT_Eltwise: {
                U32 bytes = opsPointer[i].ps.eltwise_spec.elt_sum_spec.coeff_size * sizeof(float);
                memcpy(pointer4parameterSpecs, opsPointer[i].ps.eltwise_spec.elt_sum_spec.coeff_values, bytes);
                pointer4parameterSpecs += bytes;
                break;
            }
            default:
                break;
        }
        pointer4opsName = (I8 *)pointer4parameterSpecs;
    }

    tmp->clear();
    CHECK_REQUIREMENT(pointer4opsName - data == bufSize);
    tmp->assign(data, data + bufSize);
    delete [] data;
    return SUCCESS;
}

EE deserialize_operator(const char* bytes, ModelSpec* spec, U32* pos)
{
    const char* pointer = bytes + *pos;
    I32* p4numOperatorSpecs = (I32 *)pointer;
    spec->num_operator_specs = *p4numOperatorSpecs;
    pointer += sizeof(U32);
    *pos += sizeof(U32);

    OperatorSpec *ptr = (OperatorSpec*)mt_new_storage(spec->num_operator_specs * sizeof(OperatorSpec));
    spec->ops = ptr;
    for (int i = 0; i < spec->num_operator_specs; i++) {
        str_copy(ptr[i].name, pointer, NAME_LEN);
        pointer += NAME_LEN * sizeof(I8);
        *pos += NAME_LEN * sizeof(I8);

        ptr[i].type = *((OperatorType *)pointer);
        pointer += sizeof(OperatorType);
        *pos += sizeof(OperatorType);

        ptr[i].num_inputs = *((U32 *)pointer);
        pointer += sizeof(U32);
        *pos += sizeof(U32);

        ptr[i].input_tensors_name = (I8 **)mt_new_storage(ptr[i].num_inputs * sizeof(I8 *));
        for (U32 j = 0; j<ptr[i].num_inputs; j++) {
            ptr[i].input_tensors_name[j] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
            str_copy(ptr[i].input_tensors_name[j], pointer, NAME_LEN);
            pointer += NAME_LEN * sizeof(I8);
            *pos += NAME_LEN * sizeof(I8);
        }

        ptr[i].num_outputs = *((U32 *)pointer);
        pointer += sizeof(U32);
        *pos += sizeof(U32);

        ptr[i].output_tensors_name = (I8 **)mt_new_storage(ptr[i].num_outputs * sizeof(I8 *));
        for (U32 j = 0; j < ptr[i].num_outputs; j++) {
            ptr[i].output_tensors_name[j] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
            str_copy(ptr[i].output_tensors_name[j], pointer, NAME_LEN);
            pointer += NAME_LEN * sizeof(I8);
            *pos += NAME_LEN * sizeof(I8);
        }

        U32 numTensors = ptr[i].num_inputs + ptr[i].num_outputs;
        ptr[i].tensor_positions = (I32*)mt_new_storage(numTensors * sizeof(I32));
        memcpy(ptr[i].tensor_positions, pointer, numTensors * sizeof(I32));
        pointer += numTensors * sizeof(I32);
        *pos += numTensors * sizeof(I32);

        ptr[i].ps = *((ParameterSpec*)pointer);
        pointer += sizeof(ParameterSpec);
        *pos += sizeof(ParameterSpec);
        switch (ptr[i].type) {
            case OT_Eltwise: {
                U32 bytes = ptr[i].ps.eltwise_spec.elt_sum_spec.coeff_size * sizeof(float);
                ptr[i].ps.eltwise_spec.elt_sum_spec.coeff_values = (float *)mt_new_storage(bytes);
                memcpy(ptr[i].ps.eltwise_spec.elt_sum_spec.coeff_values, pointer, bytes);
                pointer += bytes;
                *pos += bytes;
                break;
            }
            default:
                break;
        }
    }

    return SUCCESS;
}

#ifdef _USE_INT8
// put off till the scheme well-marked
INT8* f16weight2int8(F16* original_data, U32 len, F16* scale)
{
    INT8* result = (INT8*)mt_new_storage(sizeof(INT8) * len);
    F16 maxWeight = -2;
    F16 minWeight = 2;
    for (U32 i = 0; i < len; i++) {
        F16 tmpWeight = original_data[i];
        if (tmpWeight > maxWeight) {
            maxWeight = tmpWeight;
        }

        if (tmpWeight < minWeight) {
            minWeight = tmpWeight;
        }
    }

    F16 perInterval = (maxWeight - minWeight) / 256;
    *scale = maxWeight / 127; 
    for (U32 i = 0; i < len; i++) {
        int hitIntervalId = (int)((original_data[i] - minWeight) / perInterval);
        int targetInt = -128 + hitIntervalId;
        result[i] = (INT8)targetInt;
    }
    return result;
}
#endif


EE serialize_weights(const ModelSpec* spec, std::string* tmp) {
    WeightSpec* tmpPointer = spec->ws;
    U32 bufSize = sizeof(I32);
    U32 weightCount = 0;
    for (int i = 0; i < spec->num_weight_specs; i++) {
        if (DeprecatedOPOptimizer::isDeprecatedOpWeight(spec, i)) {
            continue;
        }
        if (tmpPointer[i].mdt == DT_BIN01 || tmpPointer[i].mdt == DT_BIN11) {
            bufSize += sizeof(I8) * NAME_LEN + sizeof(U32) * 4 + tmpPointer[i].bytes_of_weight + tmpPointer[i].bytes_of_vec;
        } else {
            // sizeof(U32) * 4 : leninfo + mdt + bytes_of_weights + bytes_of_vec
            bufSize += sizeof(I8) * NAME_LEN + sizeof(U32) * 4 + sizeof(U8) * ((tmpPointer[i].bytes_of_weight / bytesOf(tmpPointer[i].mdt) + 1) * bytesOf(tmpPointer[i].mdt)
                   + (tmpPointer[i].bytes_of_vec / bytesOf(tmpPointer[i].mdt) + 1) * bytesOf(tmpPointer[i].mdt));
        }    
        weightCount++;
    }
    char* data = (char*)mt_new_storage(bufSize);

    I32* pointer4numWeightSpecs = (I32*)data;
    *pointer4numWeightSpecs = weightCount;
    pointer4numWeightSpecs++;

    WeightSpec* wsPointer = spec -> ws;
    char* pointer4wsOpName = (char*)pointer4numWeightSpecs;
    for (int i = 0; i < spec->num_weight_specs; i++) {
        if (DeprecatedOPOptimizer::isDeprecatedOpWeight(spec, i)) {
            continue;
        }

        U32* length = (U32*)pointer4wsOpName;
        U32 len;
        U32 isBNN = 0;
        if (wsPointer[i].mdt == DT_BIN01 || wsPointer[i].mdt == DT_BIN11) {
            isBNN = 1;
            len = wsPointer[i].bytes_of_weight + wsPointer[i].bytes_of_vec;
        } else {
            len = sizeof(U8) * ((wsPointer[i].bytes_of_weight / bytesOf(wsPointer[i].mdt) + 1) * bytesOf(wsPointer[i].mdt)
                                 + (wsPointer[i].bytes_of_vec / bytesOf(wsPointer[i].mdt) + 1) * bytesOf(wsPointer[i].mdt));
        }
        *length = len;
        pointer4wsOpName += sizeof(U32);

        str_copy(pointer4wsOpName, wsPointer[i].op_name, NAME_LEN);
        pointer4wsOpName += NAME_LEN;

        U32* pointer4wsMdt = (U32*)pointer4wsOpName;
        *pointer4wsMdt = wsPointer[i].mdt;
        pointer4wsMdt++;

        U32* pointer4wsBytesOfWeight = (U32*)pointer4wsMdt;
        *pointer4wsBytesOfWeight = wsPointer[i].bytes_of_weight;
        pointer4wsBytesOfWeight++;

        U8* pointer4wsWeight = (U8*)pointer4wsBytesOfWeight;
        memcpy(pointer4wsWeight, wsPointer[i].weight, wsPointer[i].bytes_of_weight);
        if (isBNN == 1) {
            pointer4wsWeight += wsPointer[i].bytes_of_weight;
        } else {
            pointer4wsWeight += (wsPointer[i].bytes_of_weight / bytesOf(wsPointer[i].mdt) + 1) * bytesOf(wsPointer[i].mdt);  // need to consider the memory alignmemt, such as 22
        }        

        U32* pointer4wsBytesOfVec = (U32*)pointer4wsWeight;
        *pointer4wsBytesOfVec = wsPointer[i].bytes_of_vec;
        pointer4wsBytesOfVec++;

        U8* pointer4wsVec = (U8*)pointer4wsBytesOfVec;
        memcpy(pointer4wsVec, wsPointer[i].vec, wsPointer[i].bytes_of_vec);
        if (isBNN == 1) {
            pointer4wsVec += wsPointer[i].bytes_of_vec;
        } else {
            pointer4wsVec +=  (wsPointer[i].bytes_of_vec / bytesOf(wsPointer[i].mdt) + 1) * bytesOf(wsPointer[i].mdt);
        }

        pointer4wsOpName = (char*)pointer4wsVec;
    }

    tmp->clear();
    CHECK_REQUIREMENT(pointer4wsOpName - data == bufSize);
    tmp->assign(data, data + bufSize);
    delete [] data;
    return SUCCESS;
}

EE deserialize_weight(const char* bytes, ModelSpec* spec, U32* pos) {
    const char* pointer = bytes + *pos;
    I32* p4numWeightSpecs = (I32*)pointer;
    spec->num_weight_specs = *p4numWeightSpecs;
    pointer += sizeof(U32);
    *pos += sizeof(U32);

    WeightSpec* ptr = (WeightSpec*)mt_new_storage(spec->num_weight_specs * sizeof(WeightSpec));
    spec->ws = ptr;
    for (int i = 0; i < spec->num_weight_specs; i++) {
        U32* length = (U32*)pointer;
        pointer += sizeof(U32);
        *pos += sizeof(U32);
        U32 weightBiasBytes = 0;

        str_copy(ptr[i].op_name, pointer, NAME_LEN);
        pointer += NAME_LEN;
        *pos += NAME_LEN;

        memcpy(&(ptr[i].mdt), pointer, sizeof(DataType));
        pointer += sizeof(U32);
        *pos += sizeof(U32);

        memcpy(&(ptr[i].bytes_of_weight), pointer, sizeof(U32));
        pointer += sizeof(U32);
        *pos += sizeof(U32);

        U8* ppp3 = (U8*)mt_new_storage(ptr[i].bytes_of_weight);
        memcpy(ppp3, pointer, ptr[i].bytes_of_weight);
        ptr[i].weight = ppp3;
        U32 alignSize;
        I32 isBNN = 0;
        if (ptr[i].mdt == DT_BIN01 || ptr[i].mdt == DT_BIN11) {
            isBNN = 1;
            alignSize = ptr[i].bytes_of_weight;
        } else {
            alignSize = (ptr[i].bytes_of_weight / bytesOf(ptr[i].mdt) + 1) * bytesOf(ptr[i].mdt);
        }
        
        pointer += alignSize;
        *pos += alignSize;
        weightBiasBytes += alignSize;

        memcpy(&(ptr[i].bytes_of_vec), pointer, sizeof(U32));
        pointer += sizeof(U32);
        *pos += sizeof(U32);

        U8* ppp4 = (U8*)mt_new_storage(ptr[i].bytes_of_vec);
        memcpy(ppp4, pointer, ptr[i].bytes_of_vec);
        ptr[i].vec = ppp4;
        if (isBNN == 1) {
            alignSize = ptr[i].bytes_of_vec;
        } else {
            alignSize = (ptr[i].bytes_of_vec / bytesOf(ptr[i].mdt) + 1) * bytesOf(ptr[i].mdt);
        }
        
        pointer += alignSize;
        *pos += alignSize;
        weightBiasBytes += alignSize;
       
        CHECK_REQUIREMENT(*length == weightBiasBytes);
    }
    return SUCCESS;
}


EE serialize_model(const ModelSpec* spec, std::string* bytes) {
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


EE write_to_file(std::string* bytes, const char* fn) {
    std::ofstream out(fn);
    out << *bytes;
    out.close();
    return SUCCESS;
}


EE serialize_model_to_file(const ModelSpec* spec, const char* fn) {
    std::string bytes = "";
    CHECK_STATUS(serialize_model(spec, &bytes));
    CHECK_STATUS(write_to_file(&bytes, fn));
    return SUCCESS;
}


EE deserialize_model(const char* bytes, ModelSpec* spec) {
    U32 pos = 0;
    CHECK_STATUS(deserialize_header(bytes, spec, &pos));
    CHECK_STATUS(deserialize_operator(bytes, spec, &pos));
    CHECK_STATUS(deserialize_weight(bytes, spec, &pos));
    CHECK_STATUS(opeator_relationship(spec));
    return SUCCESS;
}


EE read_from_file(const char* fn, std::string* bytes) {
    std::string inputFilePath = fn;
    std::string resultStr = "";
    std::ifstream f(inputFilePath);
    if (f) {
        std::ostringstream ss;
        ss << f.rdbuf();
        resultStr = ss.str();
    }
    if (resultStr.size() == 0) {
        std::cerr << "[ERROR] load null file " << fn << std::endl;
        exit(1);
    }
    *bytes += resultStr;
    return SUCCESS;
}


EE deserialize_model_from_file(const char* fn, ModelSpec* spec) {
    std::string bytes = "";
    CHECK_STATUS(read_from_file(fn, &bytes));
    CHECK_STATUS(deserialize_model(bytes.c_str(), spec));
    return SUCCESS;
}


EE str_copy(I8* dst, const I8* src, I32 srcLen) {
    memset(dst, 0, NAME_LEN);
    I32 copyLen = NAME_LEN - 1;
    if (copyLen > srcLen)
        copyLen = srcLen;
    memcpy(dst, src, copyLen*sizeof(I8));
    return SUCCESS;
}


void* mt_new_storage(size_t size)
{
    if (size == 0) {
        return nullptr;
    }
    else {
        U8* s = new U8[size];
        return (void*)s;
    }
}


template<typename T>
EE ws_datatype_converter(U8* originalPtr, U8* targetPtr, int paramNum) {
    F32* f32PtrParam = (F32*)originalPtr;
    T* targetPtrParam = (T*)targetPtr;
    for (int j = 0; j < paramNum; j++) {
        F32 originalParam = f32PtrParam[j];
        T changedParam = (T)originalParam;
        targetPtrParam[j] = changedParam;
    }
    return SUCCESS;
}


EE ws_datatype_converter_bnn(U8* originalPtr, U8* targetPtr, int paramNum) {
    F32* f32PtrParam = (F32*)originalPtr;
    BIN8* targetPtrParam = (BIN8*)targetPtr;
    for (int i = 0; i < paramNum; i+=8) {
        BIN8 temp = 0; // Initialize all bits to 0
        for (int j = 0; j < 8; j++) {
            U32 bitNo = 7 - j;
            if (f32PtrParam[i + j] == 1.0) { // Set bit if weight is 1.0. Works for both DOREFA and XNOR
                temp |= (1 << bitNo); 
            }
        }
        targetPtrParam[i/8] = temp;
    }
    return SUCCESS;
}


inline EE getTargetDataType(DataConvertType convertMode, DataType *type) {
    if (*type != DT_F32)
        return SUCCESS;

    switch (convertMode) {
        case F32_to_F32:{
            *type = DT_F32;
            break;
        }
        case F32_to_F16:{
            *type = DT_F16;
            break;
        }
        case F32_to_I8:{
            *type = DT_I8;
            break;
        }
        default:
            return NOT_SUPPORTED;
    }
    return SUCCESS;
}


EE ms_datatype_converter(ModelSpec* originalMs, ModelSpec* targetMs, DataConvertType convertMode)
{
    str_copy(targetMs->model_name, originalMs->model_name, NAME_LEN);
    targetMs->dt = originalMs->dt;
    CHECK_STATUS(getTargetDataType(convertMode, &(targetMs->dt)));

    targetMs->num_inputs = originalMs->num_inputs;
    targetMs->input_names = (I8 **)mt_new_storage(targetMs->num_inputs * sizeof(I8 *));
    for (I32 j = 0; j < targetMs->num_inputs; j++) {
        targetMs->input_names[j] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
        str_copy(targetMs->input_names[j], originalMs->input_names[j], NAME_LEN);
    }
    targetMs->input_dims = (TensorDesc*)mt_new_storage(targetMs->num_inputs * sizeof(TensorDesc));
    memcpy(targetMs->input_dims, originalMs->input_dims, targetMs->num_inputs * sizeof(TensorDesc));
    for (I32 i = 0; i < targetMs->num_inputs; i++) {
        CHECK_STATUS(getTargetDataType(convertMode, &(targetMs->input_dims[i].dt)));
    }

    targetMs->num_outputs = originalMs->num_outputs;
    targetMs->output_names = (I8 **)mt_new_storage(targetMs->num_outputs * sizeof(I8 *));
    for (int j = 0; j < targetMs->num_outputs; j++) {
        targetMs->output_names[j] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
        str_copy(targetMs->output_names[j], originalMs->output_names[j], NAME_LEN);
    }

    targetMs->num_operator_specs = originalMs->num_operator_specs;
    OperatorSpec* opsPtr = (OperatorSpec*)mt_new_storage(targetMs->num_operator_specs * sizeof(OperatorSpec));
    std::map<std::string, DataType> weightDataTypeMap;
    for (int i = 0; i < targetMs->num_operator_specs; i++) {
        str_copy(opsPtr[i].name, originalMs->ops[i].name, NAME_LEN);
        opsPtr[i].type = originalMs->ops[i].type;
        opsPtr[i].num_inputs = originalMs->ops[i].num_inputs;
        opsPtr[i].input_tensors_name = (I8 **)mt_new_storage(opsPtr[i].num_inputs * sizeof(I8 *));
        for (U32 j = 0; j < opsPtr[i].num_inputs; j++) {
            opsPtr[i].input_tensors_name[j] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
            memcpy(opsPtr[i].input_tensors_name[j], originalMs->ops[i].input_tensors_name[j], NAME_LEN);
        }
        opsPtr[i].num_outputs = originalMs->ops[i].num_outputs;
        opsPtr[i].output_tensors_name = (I8 **)mt_new_storage(opsPtr[i].num_outputs * sizeof(I8 *));
        for (U32 j = 0; j < opsPtr[i].num_outputs; j++) {
            opsPtr[i].output_tensors_name[j] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
            memcpy(opsPtr[i].output_tensors_name[j], originalMs->ops[i].output_tensors_name[j], NAME_LEN);
        }

        if (OT_None != opsPtr[i].type) {
            U32 numTensors = opsPtr[i].num_inputs + opsPtr[i].num_outputs;
            opsPtr[i].tensor_positions = (I32*)mt_new_storage(numTensors * sizeof(I32));
            memcpy(opsPtr[i].tensor_positions, originalMs->ops[i].tensor_positions, numTensors * sizeof(I32));
        } else {
            opsPtr[i].tensor_positions = nullptr;
        }

        opsPtr[i].ps = originalMs->ops[i].ps;

        switch (opsPtr[i].type) {
            case OT_Eltwise: {
                U32 bytes = opsPtr[i].ps.eltwise_spec.elt_sum_spec.coeff_size * sizeof(float);
                opsPtr[i].ps.eltwise_spec.elt_sum_spec.coeff_values = (float *)mt_new_storage(bytes);
                memcpy(opsPtr[i].ps.eltwise_spec.elt_sum_spec.coeff_values,
                       originalMs->ops[i].ps.eltwise_spec.elt_sum_spec.coeff_values, bytes);
                break;
            }
            case OT_SharedWeight: {
                weightDataTypeMap[opsPtr[i].name] = opsPtr[i].ps.shared_weight_spec.desc.dt;
                CHECK_STATUS(getTargetDataType(convertMode, &(opsPtr[i].ps.shared_weight_spec.desc.dt)));
                break;
            }
            case OT_PreAllocatedMemory: {
                CHECK_STATUS(getTargetDataType(convertMode, &(opsPtr[i].ps.preallocated_memory_spec.desc.dt)));
                break;
            }
            default:
                break;
        }
    }
    targetMs->ops = opsPtr;
    targetMs->num_weight_specs = originalMs->num_weight_specs;
    WeightSpec* wsPtr = (WeightSpec*)mt_new_storage(targetMs->num_weight_specs * sizeof(WeightSpec));
    for (int i = 0; i<targetMs->num_weight_specs; i++) {
        str_copy(wsPtr[i].op_name, originalMs->ws[i].op_name, NAME_LEN);

        int weightNum = 0;
        if (originalMs->ws[i].mdt == DT_BIN01 || originalMs->ws[i].mdt == DT_BIN11) {
            wsPtr[i].mdt = originalMs->ws[i].mdt;
            weightNum = originalMs->ws[i].bytes_of_weight / bytesOf(DT_F32);
            wsPtr[i].bytes_of_weight = weightNum * bytesOf(wsPtr[i].mdt) / 8;
        } else {
            DataType wdt = originalMs->ws[i].mdt;
            if (weightDataTypeMap.find(wsPtr[i].op_name) != weightDataTypeMap.end()) {
                wdt = weightDataTypeMap[wsPtr[i].op_name];
            }
            CHECK_STATUS(getTargetDataType(convertMode, &wdt));
            wsPtr[i].mdt = wdt;
            weightNum = originalMs->ws[i].bytes_of_weight / bytesOf(originalMs->ws[i].mdt);
            wsPtr[i].bytes_of_weight = weightNum * bytesOf(wsPtr[i].mdt);
        }
        wsPtr[i].weight = (U8*)mt_new_storage(wsPtr[i].bytes_of_weight);

        DataType vdt = DT_F32;
        int biasNum = originalMs->ws[i].bytes_of_vec / bytesOf(DT_F32);
        CHECK_STATUS(getTargetDataType(convertMode, &vdt));
        wsPtr[i].bytes_of_vec = biasNum * bytesOf(vdt);
        wsPtr[i].vec = (U8*)mt_new_storage(wsPtr[i].bytes_of_vec);

        switch (wsPtr[i].mdt) {
            case DT_F32: {
                CHECK_STATUS(ws_datatype_converter<F32>(originalMs->ws[i].weight, wsPtr[i].weight, weightNum));
                CHECK_STATUS(ws_datatype_converter<F32>(originalMs->ws[i].vec, wsPtr[i].vec, biasNum));
                break;
            }
            case DT_I32: {
                CHECK_STATUS(ws_datatype_converter<I32>(originalMs->ws[i].weight, wsPtr[i].weight, weightNum));
                CHECK_STATUS(ws_datatype_converter<I32>(originalMs->ws[i].vec, wsPtr[i].vec, biasNum));
                break;
            }
            case DT_U32: {
                CHECK_STATUS(ws_datatype_converter<U32>(originalMs->ws[i].weight, wsPtr[i].weight, weightNum));
                CHECK_STATUS(ws_datatype_converter<U32>(originalMs->ws[i].vec, wsPtr[i].vec, biasNum));
                break;
            }
#ifdef _USE_FP16
            case DT_F16: {
                CHECK_STATUS(ws_datatype_converter<F16>(originalMs->ws[i].weight, wsPtr[i].weight, weightNum));
                CHECK_STATUS(ws_datatype_converter<F16>(originalMs->ws[i].vec, wsPtr[i].vec, biasNum));
                break;
            }
#endif
            case DT_I8: {
                CHECK_STATUS(ws_datatype_converter<I8>(originalMs->ws[i].weight, wsPtr[i].weight, weightNum));
                CHECK_STATUS(ws_datatype_converter<I8>(originalMs->ws[i].vec, wsPtr[i].vec, biasNum));
                break;
            }
#ifdef _USE_FP16
            case DT_BIN01: {
                CHECK_STATUS(ws_datatype_converter_bnn(originalMs->ws[i].weight, wsPtr[i].weight, weightNum));
                CHECK_STATUS(ws_datatype_converter<F16>(originalMs->ws[i].vec, wsPtr[i].vec, biasNum)); // Assume F16 for the vector
                break;
            }
            case DT_BIN11: {
                CHECK_STATUS(ws_datatype_converter_bnn(originalMs->ws[i].weight, wsPtr[i].weight, weightNum));
                CHECK_STATUS(ws_datatype_converter<F16>(originalMs->ws[i].vec, wsPtr[i].vec, biasNum)); // Assume F16 for the vector
                break;
            }
#endif
            default:
                return NOT_SUPPORTED;
        }
    }
    targetMs->ws = wsPtr;

    if (nullptr != originalMs->op_relationship_entries) {
        targetMs->num_op_tensor_entries = originalMs->num_op_tensor_entries;
        targetMs->op_relationship_entries = (OperatorRelationshipMapEntry *)mt_new_storage(targetMs->num_op_tensor_entries * sizeof(OperatorRelationshipMapEntry));
        for (int i = 0; i < targetMs->num_op_tensor_entries; i++) {
           str_copy(targetMs->op_relationship_entries[i].op, originalMs->op_relationship_entries[i].op, NAME_LEN);

           targetMs->op_relationship_entries[i].num_inputs = originalMs->op_relationship_entries[i].num_inputs;
           targetMs->op_relationship_entries[i].input_op_names = (I8 **)mt_new_storage(targetMs->op_relationship_entries[i].num_inputs * sizeof(I8 *));
           for (U32 j = 0; j < targetMs->op_relationship_entries[i].num_inputs; j++) {
              targetMs->op_relationship_entries[i].input_op_names[j] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
              str_copy(targetMs->op_relationship_entries[i].input_op_names[j], originalMs->op_relationship_entries[i].input_op_names[j], NAME_LEN);
           }

           targetMs->op_relationship_entries[i].num_outputs = originalMs->op_relationship_entries[i].num_outputs; 
           targetMs->op_relationship_entries[i].output_op_names = (I8 **)mt_new_storage(targetMs->op_relationship_entries[i].num_outputs * sizeof(I8 *));
           for (U32 j = 0; j < targetMs->op_relationship_entries[i].num_outputs; j++) {
              targetMs->op_relationship_entries[i].output_op_names[j] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
              str_copy(targetMs->op_relationship_entries[i].output_op_names[j], originalMs->op_relationship_entries[i].output_op_names[j], NAME_LEN);
           }
        }
    } else {
        targetMs->num_op_tensor_entries = 0;
        targetMs->op_relationship_entries = nullptr;
    }
    return SUCCESS;
}
