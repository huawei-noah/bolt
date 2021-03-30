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
#include <map>
#include <set>
#include <vector>
#ifndef _WIN32
#include <sys/mman.h>
#include <fcntl.h>
#endif
#include <sys/stat.h>

#include "model_common.h"
#include "profiling.h"

EE operator_relationship(ModelSpec *spec)
{
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

        if (spec->ops[i].type == OT_None) {
            continue;
        }

        // dealing with the relationship of  op -- input tensors
        for (int j = 0; j < in_tensor_number; j++) {
            std::string tmpInTensor = spec->ops[i].input_tensors_name[j];
            if (inplaceTensors.find(tmpInTensor) != inplaceTensors.end()) {  // judge inplace op or not
                int inId;
                if (inplaceTensorInNum.find(tmpInTensor) == inplaceTensorInNum.end()) {
                    inId = 1;
                    inplaceTensorInNum.insert(std::make_pair(tmpInTensor, inId));
                    opCanInChange[tmpInTensor] = true;
                } else {
                    if (opCanInChange[tmpInTensor] == false) {
                        inId = inplaceTensorInNum[tmpInTensor] + 1;
                        // inplaceTensorInNum.insert(std::make_pair(tmpInTensor, inId));
                        inplaceTensorInNum[tmpInTensor] = inId;
                        opCanInChange[tmpInTensor] = true;
                    } else {
                        inId = inplaceTensorInNum[tmpInTensor];
                        opCanInChange[tmpInTensor] = true;
                    }
                }
                std::string tmpInTensorChanged = tmpInTensor + "_" + std::to_string(inId);
                inTensorVec.push_back(tmpInTensorChanged);

                if (tensorFlowsToOpSet.find(tmpInTensorChanged) == tensorFlowsToOpSet.end()) {
                    std::vector<std::string> tmpVector;
                    tmpVector.push_back(currentOpName);
                    tensorFlowsToOpSet.insert(std::make_pair(tmpInTensorChanged, tmpVector));
                } else {
                    tensorFlowsToOpSet[tmpInTensorChanged].push_back(currentOpName);
                }

            } else {
                inTensorVec.push_back(tmpInTensor);

                if (tensorFlowsToOpSet.find(tmpInTensor) == tensorFlowsToOpSet.end()) {
                    std::vector<std::string> tmpVector;
                    tmpVector.push_back(currentOpName);
                    tensorFlowsToOpSet.insert(std::make_pair(tmpInTensor, tmpVector));
                } else {
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
            } else {
                outId = inplaceTensorOutNum[tmpOutTensor] + 1;
                // inplaceTensorOutNum.insert(std::make_pair(tmpOutTensor, outId)); can not update
                inplaceTensorOutNum[tmpOutTensor] = outId;
                opCanInChange[tmpOutTensor] = false;
            }
            std::string tmpOutTensorChanged = tmpOutTensor + "_" + std::to_string(outId);
            opOutTensorNew.insert(std::make_pair(currentOpName, tmpOutTensorChanged));
            tensorOpMapping.insert(std::make_pair(tmpOutTensorChanged, currentOpName));
        } else {
            opOutTensorNew.insert(std::make_pair(currentOpName, tmpOutTensor));
            tensorOpMapping.insert(std::make_pair(tmpOutTensor, currentOpName));
        }
    }

    // assign op-op relationship
    int opNum = spec->num_operator_specs;
    spec->num_op_tensor_entries = opNum;
    OperatorSpec *opsPtr2 = spec->ops;
    OperatorRelationshipMapEntry *oprmePtr = (OperatorRelationshipMapEntry *)mt_new_storage(
        sizeof(OperatorRelationshipMapEntry) * opNum);
    spec->op_relationship_entries = oprmePtr;
    for (int j = 0; j < opNum; j++) {
        str_copy(oprmePtr[j].op, opsPtr2[j].name, NAME_LEN);
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

template <DataType dt, typename T>
void dequantize_int8_weight(int num, F32 scale, INT8 *q, T *d)
{
    F32 factor = 1 / scale;
    T table[255];
    int base = -127;
    for (int i = 0; i < 255; i++) {
        F32 value = factor * base;
#ifndef __aarch64__
        if (dt != DT_F16) {
#endif
            table[i] = value;
#ifndef __aarch64__
        } else {
            transformFromFloat(DT_F16, &value, table + i, 1);
        }
#endif
        base++;
    }
    T *mid = table + 127;
    for (int i = 0; i < num; i++) {
        d[i] = *(mid + q[i]);
    }
}

template <typename T>
inline void deserialize_field(const char **buffer, U32 *position, T *element, int length = 1)
{
    int size = length * sizeof(T);
    memcpy(element, *buffer, size);
    *buffer += size;
    *position += size;
}

EE deserialize_header(const char *bytes, ModelSpec *spec, U32 *pos)
{
    const char *header_pointer = bytes + *pos;
    const char **pointer = &header_pointer;

    deserialize_field<I32>(pointer, pos, &spec->version);
    if (spec->version != sg_boltVersion) {
        UNI_ERROR_LOG("X2bolt version is [%d], but your model version is : [%d].\n Please update "
                      "X2bolt to version[%d].\n",
            sg_boltVersion, spec->version, spec->version);
        CHECK_STATUS(NOT_MATCH);
        return NOT_MATCH;
    }

    deserialize_field<I32>(pointer, pos, &spec->magic_number);
    if (spec->magic_number != sg_magicNumber) {
        UNI_ERROR_LOG(
            "magic_number not_match: code %d bolt model %d\n", sg_magicNumber, spec->magic_number);
        CHECK_STATUS(NOT_MATCH);
        return NOT_MATCH;
    }

    deserialize_field<I8>(pointer, pos, spec->model_name, NAME_LEN);
    deserialize_field<DataType>(pointer, pos, &spec->dt);

    deserialize_field<I32>(pointer, pos, &spec->num_inputs);
    spec->input_names = (I8 **)mt_new_storage(spec->num_inputs * sizeof(I8 *));
    spec->input_dims = (TensorDesc *)mt_new_storage(spec->num_inputs * sizeof(TensorDesc));
    for (int i = 0; i < spec->num_inputs; i++) {
        spec->input_names[i] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
        deserialize_field<I8>(pointer, pos, spec->input_names[i], NAME_LEN);
    }
    deserialize_field<TensorDesc>(pointer, pos, spec->input_dims, spec->num_inputs);

    deserialize_field<I32>(pointer, pos, &spec->num_outputs);
    spec->output_names = (I8 **)mt_new_storage(spec->num_outputs * NAME_LEN);
    for (int i = 0; i < spec->num_outputs; i++) {
        spec->output_names[i] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
        deserialize_field<I8>(pointer, pos, spec->output_names[i], NAME_LEN);
    }
    return SUCCESS;
}

EE deserialize_operator(const char *bytes, ModelSpec *spec, U32 *pos)
{
    const char *operator_pointer = bytes + *pos;
    const char **pointer = &operator_pointer;

    deserialize_field<I32>(pointer, pos, &spec->num_operator_specs);
    spec->ops = (OperatorSpec *)mt_new_storage(spec->num_operator_specs * sizeof(OperatorSpec));
    OperatorSpec *ptr = spec->ops;
    for (int i = 0; i < spec->num_operator_specs; i++) {
        deserialize_field<I8>(pointer, pos, ptr[i].name, NAME_LEN);
        deserialize_field<OperatorType>(pointer, pos, &ptr[i].type);

        deserialize_field<U32>(pointer, pos, &ptr[i].num_inputs);
        ptr[i].input_tensors_name = (I8 **)mt_new_storage(ptr[i].num_inputs * sizeof(I8 *));
        for (U32 j = 0; j < ptr[i].num_inputs; j++) {
            ptr[i].input_tensors_name[j] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
            deserialize_field<I8>(pointer, pos, ptr[i].input_tensors_name[j], NAME_LEN);
        }

        deserialize_field<U32>(pointer, pos, &ptr[i].num_outputs);
        ptr[i].output_tensors_name = (I8 **)mt_new_storage(ptr[i].num_outputs * sizeof(I8 *));
        for (U32 j = 0; j < ptr[i].num_outputs; j++) {
            ptr[i].output_tensors_name[j] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
            deserialize_field<I8>(pointer, pos, ptr[i].output_tensors_name[j], NAME_LEN);
        }

        U32 numTensors = ptr[i].num_inputs + ptr[i].num_outputs;
        ptr[i].tensor_positions = (I32 *)mt_new_storage(numTensors * sizeof(I32));
        deserialize_field<I32>(pointer, pos, ptr[i].tensor_positions, numTensors);

        deserialize_field<U32>(pointer, pos, &ptr[i].num_quant_feature);
        ptr[i].feature_scale =
            (QuantSpec *)mt_new_storage(ptr[i].num_quant_feature * sizeof(QuantSpec));
        for (U32 j = 0; j < ptr[i].num_quant_feature; j++) {
            deserialize_field<I32>(pointer, pos, &(ptr[i].feature_scale[j].num_scale));
            ptr[i].feature_scale[j].scale =
                (F32 *)mt_new_storage(ptr[i].feature_scale[j].num_scale * sizeof(F32));
            deserialize_field<F32>(
                pointer, pos, ptr[i].feature_scale[j].scale, ptr[i].feature_scale[j].num_scale);
        }

        deserialize_field<U8>(
            pointer, pos, (U8 *)&(ptr[i].ps), get_operator_parameter_size(ptr[i].type));
    }
    return SUCCESS;
}

EE deserialize_weight(const char *bytes, ModelSpec *spec, U32 *pos)
{
    const char *weight_pointer = bytes + *pos;
    const char **pointer = &weight_pointer;

    deserialize_field<I32>(pointer, pos, &spec->num_weight_specs);
    spec->ws = (WeightSpec *)mt_new_storage(spec->num_weight_specs * sizeof(WeightSpec));
    WeightSpec *ptr = spec->ws;
    for (int i = 0; i < spec->num_weight_specs; i++) {
        U32 length = 0, count = 0;
        deserialize_field<U32>(pointer, pos, &length);

        deserialize_field<I8>(pointer, pos, ptr[i].op_name, NAME_LEN);
        deserialize_field<DataType>(pointer, pos, &ptr[i].mdt);

        bool quantFP16 = false;
        bool quantInt8 = false;
        if (DT_F16 == ptr[i].mdt && DT_F32 == spec->dt) {
            ptr[i].mdt = DT_F32;
            quantFP16 = true;
        } else if (DT_I8 == ptr[i].mdt && DT_I8 != spec->dt) {
            ptr[i].mdt = (spec->dt == DT_F16_8Q) ? DT_F16 : spec->dt;
            quantInt8 = true;
        }

        deserialize_field<U32>(pointer, pos, &ptr[i].bytes_of_weight);
        U8 *serialWeight = (U8 *)(*pointer);
        if (ptr[i].bytes_of_weight == 0) {
            serialWeight = nullptr;
        }
        *pointer += ptr[i].bytes_of_weight;
        *pos += ptr[i].bytes_of_weight;
        count += ptr[i].bytes_of_weight;
        if (quantFP16) {
            ptr[i].bytes_of_weight *= 2;
        }
        if (quantInt8) {
            ptr[i].bytes_of_weight *= bytesOf(ptr[i].mdt);
        }

        deserialize_field<U32>(pointer, pos, &ptr[i].bytes_of_vec);
        U8 *serialBias = (U8 *)(*pointer);
        if (ptr[i].bytes_of_vec == 0) {
            serialBias = nullptr;
        }
        *pointer += ptr[i].bytes_of_vec;
        *pos += ptr[i].bytes_of_vec;
        count += ptr[i].bytes_of_vec;
        if (quantFP16) {
            ptr[i].bytes_of_vec *= 2;
        }

        deserialize_field<U32>(pointer, pos, &ptr[i].num_quant_scale);
        ptr[i].weight_scale =
            (QuantSpec *)mt_new_storage(ptr[i].num_quant_scale * sizeof(QuantSpec));
        for (U32 j = 0; j < ptr[i].num_quant_scale; j++) {
            deserialize_field<I32>(pointer, pos, &(ptr[i].weight_scale[j].num_scale));
            ptr[i].weight_scale[j].scale =
                (F32 *)mt_new_storage(ptr[i].weight_scale[j].num_scale * sizeof(F32));
            deserialize_field<F32>(
                pointer, pos, ptr[i].weight_scale[j].scale, ptr[i].weight_scale[j].num_scale);
        }

        CHECK_REQUIREMENT(length == count);

        if (quantFP16) {
            ptr[i].weight = (U8 *)mt_new_storage(ptr[i].bytes_of_weight);
            ptr[i].vec = (U8 *)mt_new_storage(ptr[i].bytes_of_vec);
            transformToFloat(DT_F16, serialWeight, (F32 *)ptr[i].weight, ptr[i].bytes_of_weight / 4);
            transformToFloat(DT_F16, serialBias, (F32 *)ptr[i].vec, ptr[i].bytes_of_vec / 4);
        } else {
            if (quantInt8) {
                CHECK_REQUIREMENT(
                    1 == ptr[i].num_quant_scale && 1 == ptr[i].weight_scale[0].num_scale);
                ptr[i].weight = (U8 *)mt_new_storage(ptr[i].bytes_of_weight);
                F32 scale = ptr[i].weight_scale[0].scale[0];
                if (DT_F32 == ptr[i].mdt) {
                    dequantize_int8_weight<DT_F32, F32>(ptr[i].bytes_of_weight / 4, scale,
                        (INT8 *)serialWeight, (F32 *)ptr[i].weight);
                } else if (DT_F16 == ptr[i].mdt) {
#ifdef __aarch64__
                    dequantize_int8_weight<DT_F16, F16>(ptr[i].bytes_of_weight / 2, scale,
                        (INT8 *)serialWeight, (F16 *)ptr[i].weight);
#else
                    dequantize_int8_weight<DT_F16, unsigned short>(ptr[i].bytes_of_weight / 2,
                        scale, (INT8 *)serialWeight, (unsigned short *)ptr[i].weight);
#endif
                } else {
                    UNI_ERROR_LOG("Can not support convert INT8 data to %d.\n", ptr[i].mdt);
                    exit(1);
                }
            } else {
                ptr[i].weight = serialWeight;
            }
            ptr[i].vec = serialBias;
        }
    }
    return SUCCESS;
}

EE deserialize_model(const char *bytes, ModelSpec *spec)
{
    U32 pos = 0;
    CHECK_STATUS(deserialize_header(bytes, spec, &pos));
    CHECK_STATUS(deserialize_operator(bytes, spec, &pos));
    CHECK_STATUS(deserialize_weight(bytes, spec, &pos));
    CHECK_STATUS(operator_relationship(spec));
    if (spec->mfd->useFileStream) {
        spec->mfd->fileLength = pos;
    }
    return SUCCESS;
}

EE deserialize_model_from_file(const char *fn, ModelSpec *spec, bool useFileStream)
{
    UNI_DEBUG_LOG("Read bolt model from %s...\n", (useFileStream ? "file stream" : fn));
    UNI_PROFILE(
        {
            char *bytes = nullptr;
            int fd;
            size_t fileLength;
            spec->mfd = (ModelFileDescriptor *)mt_new_storage(sizeof(ModelFileDescriptor));
            spec->mfd->useFileStream = useFileStream;
            if (useFileStream) {
                bytes = (char *)fn;
            } else {
#ifdef _WIN32
                FILE *file = fopen(fn, "rb");
                if (file == NULL) {
                    UNI_ERROR_LOG("Cannot open bolt model file %s.\n", fn);
                    return FILE_ERROR;
                }

                fseek(file, 0, SEEK_END);
                fileLength = ftell(file);
                rewind(file);

                bytes = (char *)malloc(sizeof(char) * fileLength);
                if (bytes == NULL) {
                    UNI_ERROR_LOG("Memory allocated for model failed.\n");
                }

                size_t result = fread(bytes, 1, fileLength, file);
                if (result != fileLength) {
                    UNI_ERROR_LOG("Read bolt model file %s failed.\n", fn);
                    return FILE_ERROR;
                }
                fclose(file);
#else
                fd = open(fn, O_RDONLY);
                if (-1 == fd) {
                    UNI_ERROR_LOG("Cannot open bolt model file %s.\n", fn);
                    return FILE_ERROR;
                }

                struct stat ss;
                if (-1 == fstat(fd, &ss)) {
                    UNI_ERROR_LOG("Cannot get size from bolt model file %s descriptor.\n", fn);
                    return FILE_ERROR;
                }

                fileLength = ss.st_size;
                bytes = (char *)mmap(nullptr, fileLength, PROT_READ, MAP_SHARED, fd, 0);
                if (MAP_FAILED == bytes) {
                    UNI_ERROR_LOG("Mmap bolt model file %s failed.\n", fn);
                    return FILE_ERROR;
                }
#endif
                spec->mfd->fd = fd;
                spec->mfd->fileLength = fileLength;
            }
            spec->mfd->bytes = bytes;

            CHECK_STATUS(deserialize_model(bytes, spec));
        },
        std::string("deserialize_model_from_file"), std::string("prepare"));
    UNI_DEBUG_LOG("Read bolt model end.\n");
    return SUCCESS;
}
