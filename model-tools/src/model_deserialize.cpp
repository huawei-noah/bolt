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
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

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
    } else {
        U8* s = new U8[size];
        return (void*)s;
    }
}

EE operator_relationship(ModelSpec* spec) {
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
            if (inplaceTensors.find(tmpInTensor) != inplaceTensors.end()) {  // judge inplace op or not
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

template<typename T>
void dequantize_int8_weight(int num, F32 scale, INT8* q, T* d)
{
    F32 factor = 1 / scale;
    T table[255];
    int base = -127;
    for (int i = 0; i < 255; i++) {
        table[i] = factor * base;
        base++;
    }
    T *mid = table + 127;
    for (int i = 0; i < num; i++) {
        d[i] = *(mid + q[i]);
    }
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

        ptr[i].num_quant_feature = *((U32 *)pointer);
        pointer += sizeof(U32);
        *pos += sizeof(U32);

        if (0 != ptr[i].num_quant_feature) {
            ptr[i].feature_scale = (QuantSpec*)mt_new_storage(ptr[i].num_quant_feature * sizeof(QuantSpec));
        } else {
            ptr[i].feature_scale = nullptr;
        }
        for (U32 j = 0; j < ptr[i].num_quant_feature; j++) {
            ptr[i].feature_scale[j].num_scale = *((int *)pointer);
            int num = ptr[i].feature_scale[j].num_scale;
            pointer += sizeof(int);
            *pos += sizeof(int);

            ptr[i].feature_scale[j].scale = (F32*)mt_new_storage(num * sizeof(F32));
            memcpy(ptr[i].feature_scale[j].scale, pointer, num * sizeof(F32));
            pointer += num * sizeof(F32);
            *pos += num * sizeof(F32);
        }

        ptr[i].ps = *((ParameterSpec*)pointer);
        pointer += sizeof(ParameterSpec);
        *pos += sizeof(ParameterSpec);
        switch (ptr[i].type) {
            case OT_Eltwise: {
                if (ptr[i].ps.eltwise_spec.elt_mode == ELTWISE_SUM) {
                    U32 bytes = ptr[i].ps.eltwise_spec.elt_sum_spec.coeff_size * sizeof(float);
                    ptr[i].ps.eltwise_spec.elt_sum_spec.coeff_values = (float *)mt_new_storage(bytes);
                    memcpy(ptr[i].ps.eltwise_spec.elt_sum_spec.coeff_values, pointer, bytes);
                    pointer += bytes;
                    *pos += bytes;
                }
                break;
            }
            default:
                break;
        }
    }

    return SUCCESS;
}

EE deserialize_weight(const char* bytes, ModelSpec* spec, U32* pos)
{
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

        bool quantWeight = false;
        if (DT_I8 == ptr[i].mdt && DT_I8 != spec->dt) {
            ptr[i].mdt = (spec->dt == DT_F16_8Q) ? DT_F16 : spec->dt;
            quantWeight = true;
        }

        memcpy(&(ptr[i].bytes_of_weight), pointer, sizeof(U32));
        U32 alignSize = ptr[i].bytes_of_weight;
        if (quantWeight) {
            ptr[i].bytes_of_weight *= bytesOf(ptr[i].mdt);
        }
        pointer += sizeof(U32);
        *pos += sizeof(U32);

        ptr[i].weight = (U8*)mt_new_storage(ptr[i].bytes_of_weight);
        INT8 *serialWeight = (INT8*)pointer;

        pointer += alignSize;
        *pos += alignSize;
        weightBiasBytes += alignSize;

        memcpy(&(ptr[i].bytes_of_vec), pointer, sizeof(U32));
        pointer += sizeof(U32);
        *pos += sizeof(U32);

        U8* ppp4 = (U8*)mt_new_storage(ptr[i].bytes_of_vec);
        memcpy(ppp4, pointer, ptr[i].bytes_of_vec);
        ptr[i].vec = ppp4;
        
        pointer += ptr[i].bytes_of_vec;
        *pos += ptr[i].bytes_of_vec;
        weightBiasBytes += ptr[i].bytes_of_vec;

        memcpy(&(ptr[i].num_quant_scale), pointer, sizeof(U32));
        pointer += sizeof(U32);
        *pos += sizeof(U32);

        if (0 != ptr[i].num_quant_scale) {
            ptr[i].weight_scale = (QuantSpec*)mt_new_storage(ptr[i].num_quant_scale * sizeof(QuantSpec));
        }
        for (U32 j = 0; j < ptr[i].num_quant_scale; j++) {
            ptr[i].weight_scale[j].num_scale = *((int *)pointer);
            int num = ptr[i].weight_scale[j].num_scale;
            pointer += sizeof(int);
            *pos += sizeof(int);

            ptr[i].weight_scale[j].scale = (F32*)mt_new_storage(num * sizeof(F32));
            memcpy(ptr[i].weight_scale[j].scale, pointer, num * sizeof(F32));
            pointer += num * sizeof(F32);
            *pos += num * sizeof(F32);
        }
       
        CHECK_REQUIREMENT(*length == weightBiasBytes);
        
        if (quantWeight) {
            CHECK_REQUIREMENT(1 == ptr[i].num_quant_scale && 1 == ptr[i].weight_scale[0].num_scale);
            F32 scale = ptr[i].weight_scale[0].scale[0];
            if (DT_F32 == ptr[i].mdt) {
                dequantize_int8_weight<F32>(alignSize, scale, serialWeight, (F32*)ptr[i].weight);
            } else {
#ifdef __aarch64__
                dequantize_int8_weight<F16>(alignSize, scale, serialWeight, (F16*)ptr[i].weight);
#endif
            }
        } else {
            memcpy(ptr[i].weight, serialWeight, ptr[i].bytes_of_weight);
        }
    }
    return SUCCESS;
}

EE deserialize_model(const char* bytes, ModelSpec* spec)
{
    U32 pos = 0;
    CHECK_STATUS(deserialize_header(bytes, spec, &pos));
    CHECK_STATUS(deserialize_operator(bytes, spec, &pos));
    CHECK_STATUS(deserialize_weight(bytes, spec, &pos));
    CHECK_STATUS(operator_relationship(spec));
    return SUCCESS;
}

int read_from_file(const char* fn, char** bytes)
{
    int fd = open(fn, O_RDONLY);
    CHECK_REQUIREMENT(-1 != fd);

    struct stat ss;
    CHECK_REQUIREMENT(fstat(fd, &ss) != -1);
    
    int fileLength = ss.st_size;
    *bytes = (char*)mmap(nullptr, fileLength, PROT_READ,
                       MAP_SHARED, fd, 0);
    CHECK_REQUIREMENT(MAP_FAILED != bytes);
    close(fd);
    return fileLength;
}

EE deserialize_model_from_file(const char* fn, ModelSpec* spec)
{
    char *bytes = nullptr;
    int fileLength = read_from_file(fn, &bytes);
    CHECK_STATUS(deserialize_model(bytes, spec));
    munmap(bytes, fileLength);
    return SUCCESS;
}
