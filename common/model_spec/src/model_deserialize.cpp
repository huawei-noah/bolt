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
#if defined(__GLIBC__) || defined(__linux__)
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#endif

#include "model_common.h"
#include "profiling.h"
#include "thread_affinity.h"
#include "file.h"

#ifdef _USE_OP_TENSOR_RELATIONS
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
    OperatorRelationshipMapEntry *oprmePtr =
        (OperatorRelationshipMapEntry *)mt_malloc(sizeof(OperatorRelationshipMapEntry) * opNum);
    spec->op_relationship_entries = oprmePtr;
    for (int j = 0; j < opNum; j++) {
        str_copy(oprmePtr[j].op, opsPtr2[j].name, NAME_LEN);
        int opInOpNum = opInTensorNew[opsPtr2[j].name].size();
        oprmePtr[j].num_inputs = opInOpNum;
        oprmePtr[j].input_op_names = (I8 **)mt_malloc(opInOpNum * sizeof(I8 *));
        for (int k = 0; k < opInOpNum; k++) {
            oprmePtr[j].input_op_names[k] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
            std::string ten_name = opInTensorNew[opsPtr2[j].name][k];
            std::string tensor2op = tensorOpMapping[ten_name];
            str_copy(oprmePtr[j].input_op_names[k], tensor2op.c_str(), tensor2op.length());
        }

        int opOutOpNum = tensorFlowsToOpSet[opOutTensorNew[opsPtr2[j].name]].size();
        oprmePtr[j].num_outputs = opOutOpNum;
        oprmePtr[j].output_op_names = (I8 **)mt_malloc(opOutOpNum * sizeof(I8 *));
        for (int k = 0; k < opOutOpNum; k++) {
            oprmePtr[j].output_op_names[k] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
            std::string tensor2op = tensorFlowsToOpSet[opOutTensorNew[opsPtr2[j].name]][k];
            str_copy(oprmePtr[j].output_op_names[k], tensor2op.c_str(), tensor2op.length());
        }
    }
    return SUCCESS;
}
#endif

template <DataType dt, typename T>
void dequantize_int8_weight(int num, F32 scale, INT8 *q, T *d)
{
    F32 factor = 1 / scale;
    T table[255];
    int base = -127;
    for (int i = 0; i < 255; i++) {
        F32 value = factor * base;
#ifndef _USE_FP16_TYPE
        if (dt != DT_F16) {
#endif
            table[i] = value;
#ifndef _USE_FP16_TYPE
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

template <DataType dt, typename T>
void dequantize_int4_weight(int num, F32 scale, INT8 *q, T *d)
{
    F32 factor = 1 / scale;
    T table[16];
    for (int i = 0; i < 15; i++) {
        F32 value = factor * (i - 7);
#ifndef _USE_FP16_TYPE
        if (dt != DT_F16) {
#endif
            table[i] = value;
#ifndef _USE_FP16_TYPE
        } else {
            transformFromFloat(DT_F16, &value, table + i, 1);
        }
#endif
    }
    T *mid = table;
    for (int i = 0; i < num; i++) {
        d[i] = *(mid + ((q[i / 2] >> ((i % 2) * 4)) & 0xF));
    }
}

void dequantize_int4_int8(int num, INT8 *q, INT8 *d)
{
    for (int i = 0; i < num; i++) {
        d[i] = ((q[i / 2] >> ((i % 2) * 4)) & 0xF) - 7;
    }
}

void dequantize_int4_u8_q(int num, INT8 *q, UINT8 *d)
{
    for (int i = 0; i < num; i++) {
        d[i] = int((q[i / 2] >> ((i % 2) * 4)) & 0xF) + 121;
    }
}

Arch updateInferDtByArch(ModelSpec *spec, DataType targetDt)
{
    Arch arch = get_cpu_arch();
    DataType dt = spec->dt;
    if ((dt == DT_I8) || (dt == DT_I4)) {
        dt = targetDt;
#ifdef _USE_X86_ARM_CONSISTENCY
        if (dt == DT_F16_8Q) {
            dt = DT_F32_8Q;
        }
#endif
    }
    if (dt == DT_F16) {
        if ((targetDt == DT_F32_8Q) || (targetDt == DT_F32)) {
            dt = DT_F32;
        }
    }
    if ((dt == DT_F32) && (targetDt == DT_F16)) {
        dt = DT_F16;
    }
    if (dt == DT_BIN01) {
        if (targetDt != DT_F16_8Q) {
            dt = DT_F32;
        } else {
            dt = DT_F16;
        }
    }

    if (dt != spec->dt) {
        std::string precision = DataTypeName()[spec->dt];
        if (precision.back() == 'Q') {
            precision = "DT_I8";
        }
        if (std::string(DataTypeName()[spec->dt]) != precision) {
            UNI_WARNING_LOG("%s is not Supported on this arch, bolt will use %s inference! \n",
                DataTypeName()[spec->dt], precision.c_str())
        }
        spec->dt = dt;
    }
    return arch;
}

void TransWeightFromBINToF32(WeightSpec *ptr, INT8 *src)
{
    ptr->bytes_of_weight *= 32;
    CHECK_REQUIREMENT((DT_BIN01 == ptr->mdt) || (DT_BIN11 == ptr->mdt));
    ptr->weight = (U8 *)mt_malloc(ptr->bytes_of_weight);
    transformToFloat(ptr->mdt, src, (F32 *)ptr->weight, ptr->bytes_of_weight / 4);
}

void TransWeightFromI4ToF32(WeightSpec *ptr, INT8 *src)
{
    ptr->bytes_of_weight *= 8;
    CHECK_REQUIREMENT(1 == ptr->num_quant_scale && 1 == ptr->weight_scale[0].num_scale);
    ptr->weight = (U8 *)mt_malloc(ptr->bytes_of_weight);
    F32 scale = ptr->weight_scale[0].scale[0];
    dequantize_int4_weight<DT_F32, F32>(
        ptr->bytes_of_weight / 4, scale, (INT8 *)src, (F32 *)(ptr->weight));
    ptr->weight_scale[0].scale[0] = 0;
}

void TransWeightFromI4ToF16(WeightSpec *ptr, INT8 *src)
{
    ptr->bytes_of_weight *= 4;
    CHECK_REQUIREMENT(1 == ptr->num_quant_scale && 1 == ptr->weight_scale[0].num_scale);
    ptr->weight = (U8 *)mt_malloc(ptr->bytes_of_weight);
    F32 scale = ptr->weight_scale[0].scale[0];
    dequantize_int4_weight<DT_F16, F16>(
        ptr->bytes_of_weight / 2, scale, (INT8 *)src, (F16 *)(ptr->weight));
    ptr->weight_scale[0].scale[0] = 0;
}

void TransWeightFromI4ToI8(WeightSpec *ptr, INT8 *src)
{
    ptr->bytes_of_weight *= 2;
    CHECK_REQUIREMENT(1 == ptr->num_quant_scale && 1 == ptr->weight_scale[0].num_scale);
    ptr->weight = (U8 *)mt_malloc(ptr->bytes_of_weight);
    F32 scale = ptr->weight_scale[0].scale[0];
    dequantize_int4_int8(ptr->bytes_of_weight, (INT8 *)src, (INT8 *)ptr->weight);
}

void TransWeightFromI4ToU8(WeightSpec *ptr, INT8 *src)
{
    ptr->bytes_of_weight *= 2;
    CHECK_REQUIREMENT(1 == ptr->num_quant_scale && 1 == ptr->weight_scale[0].num_scale);
    ptr->weight = (U8 *)mt_malloc(ptr->bytes_of_weight);
    F32 scale = ptr->weight_scale[0].scale[0];
    dequantize_int4_u8_q(ptr->bytes_of_weight, (INT8 *)src, (UINT8 *)ptr->weight);
}

void TransWeightFromI8ToF32(WeightSpec *ptr, INT8 *src)
{
    ptr->bytes_of_weight *= 4;
    CHECK_REQUIREMENT(1 == ptr->num_quant_scale && 1 == ptr->weight_scale[0].num_scale);
    ptr->weight = (U8 *)mt_malloc(ptr->bytes_of_weight);
    F32 scale = ptr->weight_scale[0].scale[0];
    dequantize_int8_weight<DT_F32, F32>(
        ptr->bytes_of_weight / 4, scale, (INT8 *)src, (F32 *)(ptr->weight));
    ptr->weight_scale[0].scale[0] = 0;
}

void TransWeightFromI8ToF16(WeightSpec *ptr, INT8 *src)
{
    ptr->bytes_of_weight *= 2;
    CHECK_REQUIREMENT(1 == ptr->num_quant_scale && 1 == ptr->weight_scale[0].num_scale);
    ptr->weight = (U8 *)mt_malloc(ptr->bytes_of_weight);
    F32 scale = ptr->weight_scale[0].scale[0];
    dequantize_int8_weight<DT_F16, F16>(
        ptr->bytes_of_weight / 2, scale, (INT8 *)src, (F16 *)(ptr->weight));
    ptr->weight_scale[0].scale[0] = 0;
}

void TransBiasFromF32ToF16(WeightSpec *ptr, F32 *src)
{
    ptr->bytes_of_vec /= 2;
    ptr->vec = (U8 *)mt_malloc(ptr->bytes_of_vec);
    transformFromFloat(DT_F16, src, ptr->vec, ptr->bytes_of_vec / 2);
}

void TransBiasFromF16ToF32(WeightSpec *ptr, F16 *src)
{
    ptr->bytes_of_vec *= 2;
    ptr->vec = (U8 *)mt_malloc(ptr->bytes_of_vec);
    transformToFloat(DT_F16, src, (F32 *)ptr->vec, ptr->bytes_of_vec / 4);
}

void TransWeightFromF32ToF16(WeightSpec *ptr, F32 *src)
{
    ptr->bytes_of_weight /= 2;
    ptr->weight = (U8 *)mt_malloc(ptr->bytes_of_weight);
    transformFromFloat(DT_F16, src, ptr->weight, ptr->bytes_of_weight / 2);
}

void TransWeightFromF16ToF32(WeightSpec *ptr, F16 *src)
{
    ptr->bytes_of_weight *= 2;
    ptr->weight = (U8 *)mt_malloc(ptr->bytes_of_weight);
    transformToFloat(DT_F16, src, (F32 *)ptr->weight, ptr->bytes_of_weight / 4);
}

template <typename T>
inline void deserialize_field(const U8 **buffer, U32 *position, T *element, int length = 1)
{
    int size = length * sizeof(T);
    UNI_MEMCPY(element, *buffer, size);
    *buffer += size;
    *position += size;
}

EE deserialize_header(const U8 *bytes, ModelSpec *spec, DataType targetDt, U32 *pos)
{
    const U8 *header_pointer = bytes + *pos;
    const U8 **pointer = &header_pointer;

    deserialize_field<I32>(pointer, pos, &spec->version);
    if (spec->version != sg_boltVersion) {
        UNI_WARNING_LOG("The read model module version(%d) of the library should match the model "
                        "file of the same version, but your model version is %d. This may "
                        "encounter error.\nPlease use another library or reconverter model.\n",
            sg_boltVersion, spec->version);
    }
    if (spec->version < 20201120) {
        UNI_ERROR_LOG("This library can not read model with version(%d),\n", spec->version);
        return NOT_MATCH;
    }

    deserialize_field<I32>(pointer, pos, &spec->magic_number);
    if (spec->magic_number != sg_magicNumber) {
        UNI_ERROR_LOG("magic number not match: library is %d, bolt model is %d\n", sg_magicNumber,
            spec->magic_number);
        return NOT_MATCH;
    }

    deserialize_field<I8>(pointer, pos, spec->model_name, NAME_LEN);
    deserialize_field<DataType>(pointer, pos, &spec->dt);
    updateInferDtByArch(spec, targetDt);

    deserialize_field<I32>(pointer, pos, &spec->num_inputs);
    spec->input_names = (I8 **)mt_malloc(spec->num_inputs * sizeof(I8 *));
    spec->input_dims = (TensorDesc *)mt_malloc(spec->num_inputs * sizeof(TensorDesc));
    for (I32 i = 0; i < spec->num_inputs; i++) {
        spec->input_names[i] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
        deserialize_field<I8>(pointer, pos, spec->input_names[i], NAME_LEN);
    }
    for (I32 i = 0; i < spec->num_inputs; i++) {
        deserialize_field<U8>(pointer, pos, (U8 *)(spec->input_dims + i),
            get_operator_parameter_size(spec->version, OT_Input));
        if ((spec->input_dims[i].dt == DT_F16) || (spec->input_dims[i].dt == DT_F32)) {
            spec->input_dims[i].dt = noQuantDataType(spec->dt);
        }
    }

    deserialize_field<I32>(pointer, pos, &spec->num_outputs);
    spec->output_names = (I8 **)mt_malloc(spec->num_outputs * NAME_LEN);
    for (I32 i = 0; i < spec->num_outputs; i++) {
        spec->output_names[i] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
        deserialize_field<I8>(pointer, pos, spec->output_names[i], NAME_LEN);
    }
    return SUCCESS;
}

EE model_compatibility(OperatorSpec *p, int version)
{
    if (version == 20201120) {
        if (p->type == OT_Conv || p->type == OT_Deconvolution) {
            p->ps.conv_spec.output_pad_t = 0;
            p->ps.conv_spec.output_pad_h = 0;
            p->ps.conv_spec.output_pad_w = 0;
        }
        if (p->type == OT_LayerNorm) {
            p->ps.ln_spec.axis = -1;
        }
    }
    if (version == 20201120 || version == 20211021) {
        if (p->type == OT_Transpose) {
            p->ps.transpose_spec.df = DF_NCHW;
        }
    }
    if (version < 20220126) {
        if (p->type == OT_Pooling) {
            p->ps.pooling_spec.count_include_pad = 0;
        }
    }
    if (version < 20220817) {
        if (p->type == OT_Resize) {
            p->ps.resize_spec.zoom_factor = 0;
            p->ps.resize_spec.pad_begin = 0;
            p->ps.resize_spec.pad_end = 0;
        }
    }
    return SUCCESS;
}

EE deserialize_operator(const U8 *bytes, ModelSpec *spec, U32 *pos)
{
    const U8 *operator_pointer = bytes + *pos;
    const U8 **pointer = &operator_pointer;
    deserialize_field<I32>(pointer, pos, &spec->num_operator_specs);
    spec->ops = (OperatorSpec *)mt_malloc(spec->num_operator_specs * sizeof(OperatorSpec));
    OperatorSpec *ptr = spec->ops;
    for (I32 i = 0; i < spec->num_operator_specs; i++) {
        deserialize_field<I8>(pointer, pos, ptr[i].name, NAME_LEN);
        deserialize_field<OperatorType>(pointer, pos, &ptr[i].type);

        deserialize_field<U32>(pointer, pos, &ptr[i].num_inputs);
        ptr[i].input_tensors_name = (I8 **)mt_malloc(ptr[i].num_inputs * sizeof(I8 *));
        for (U32 j = 0; j < ptr[i].num_inputs; j++) {
            ptr[i].input_tensors_name[j] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
            deserialize_field<I8>(pointer, pos, ptr[i].input_tensors_name[j], NAME_LEN);
        }

        deserialize_field<U32>(pointer, pos, &ptr[i].num_outputs);
        ptr[i].output_tensors_name = (I8 **)mt_malloc(ptr[i].num_outputs * sizeof(I8 *));
        for (U32 j = 0; j < ptr[i].num_outputs; j++) {
            ptr[i].output_tensors_name[j] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
            deserialize_field<I8>(pointer, pos, ptr[i].output_tensors_name[j], NAME_LEN);
        }

        U32 numTensors = ptr[i].num_inputs + ptr[i].num_outputs;
        ptr[i].tensor_positions = (I32 *)mt_malloc(numTensors * sizeof(I32));
        deserialize_field<I32>(pointer, pos, ptr[i].tensor_positions, numTensors);

        deserialize_field<U32>(pointer, pos, &ptr[i].num_quant_feature);
        ptr[i].feature_scale = (QuantSpec *)mt_malloc(ptr[i].num_quant_feature * sizeof(QuantSpec));
        for (U32 j = 0; j < ptr[i].num_quant_feature; j++) {
            deserialize_field<I32>(pointer, pos, &(ptr[i].feature_scale[j].num_scale));
            ptr[i].feature_scale[j].scale =
                (F32 *)mt_malloc(ptr[i].feature_scale[j].num_scale * sizeof(F32));
            deserialize_field<F32>(
                pointer, pos, ptr[i].feature_scale[j].scale, ptr[i].feature_scale[j].num_scale);
        }

        int param_size = get_operator_parameter_size(spec->version, ptr[i].type);
        deserialize_field<U8>(pointer, pos, (U8 *)&(ptr[i].ps), param_size);
        CHECK_STATUS(model_compatibility(ptr + i, spec->version));
        UNI_DETAIL_LOG("    op:%s type:%s size:%dB.\n", ptr[i].name, OperatorTypeName()[ptr[i].type], param_size);
    }
    return SUCCESS;
}

EE deserialize_weight(const U8 *bytes, ModelSpec *spec, U32 *pos)
{
    const U8 *weight_pointer = bytes + *pos;
    const U8 **pointer = &weight_pointer;

    deserialize_field<I32>(pointer, pos, &spec->num_weight_specs);
    spec->ws = (WeightSpec *)mt_malloc(spec->num_weight_specs * sizeof(WeightSpec));
    WeightSpec *ptr = spec->ws;
    std::map<std::string, DataType> sharedWeightDataTypeMap;
    Arch arch = get_cpu_arch();
    for (I32 i = 0; i < spec->num_operator_specs; i++) {
        if (OT_SharedWeight == spec->ops[i].type) {
            sharedWeightDataTypeMap[spec->ops[i].name] = spec->ops[i].ps.shared_weight_spec.desc.dt;
        }
    }

    for (I32 i = 0; i < spec->num_weight_specs; i++) {
        U32 length = 0, count = 0;
        deserialize_field<U32>(pointer, pos, &length);

        deserialize_field<I8>(pointer, pos, ptr[i].op_name, NAME_LEN);
        deserialize_field<DataType>(pointer, pos, &ptr[i].mdt);

        deserialize_field<U32>(pointer, pos, &ptr[i].bytes_of_weight);
        U8 *serialWeight = (U8 *)(*pointer);
        if (ptr[i].bytes_of_weight == 0) {
            serialWeight = nullptr;
        }
        *pointer += ptr[i].bytes_of_weight;
        *pos += ptr[i].bytes_of_weight;
        count += ptr[i].bytes_of_weight;

        deserialize_field<U32>(pointer, pos, &ptr[i].bytes_of_vec);
        U8 *serialBias = (U8 *)(*pointer);
        if (ptr[i].bytes_of_vec == 0) {
            serialBias = nullptr;
        }
        *pointer += ptr[i].bytes_of_vec;
        *pos += ptr[i].bytes_of_vec;
        count += ptr[i].bytes_of_vec;

        deserialize_field<U32>(pointer, pos, &ptr[i].num_quant_scale);
        ptr[i].weight_scale = (QuantSpec *)mt_malloc(ptr[i].num_quant_scale * sizeof(QuantSpec));
        for (U32 j = 0; j < ptr[i].num_quant_scale; j++) {
            deserialize_field<I32>(pointer, pos, &(ptr[i].weight_scale[j].num_scale));
            ptr[i].weight_scale[j].scale =
                (F32 *)mt_malloc(ptr[i].weight_scale[j].num_scale * sizeof(F32));
            deserialize_field<F32>(
                pointer, pos, ptr[i].weight_scale[j].scale, ptr[i].weight_scale[j].num_scale);
        }

        CHECK_REQUIREMENT(length == count);
        if (spec->dt == DT_F32) {
            if (ptr[i].mdt == DT_F16) {
                // trans w&b from 16 to 32
                TransWeightFromF16ToF32(ptr + i, (F16 *)serialWeight);
                TransBiasFromF16ToF32(ptr + i, (F16 *)serialBias);
                ptr[i].mdt = DT_F32;
            } else if (ptr[i].mdt == DT_F16_8Q) {
                // trans w from 8 to 32
                // trans b from 16 to 32
                TransWeightFromI8ToF32(ptr + i, (INT8 *)serialWeight);
                TransBiasFromF16ToF32(ptr + i, (F16 *)serialBias);
                ptr[i].mdt = DT_F32;
            } else if ((ptr[i].mdt == DT_I8) || (ptr[i].mdt == DT_F32_8Q)) {
                // trans w from 8 to 32
                TransWeightFromI8ToF32(ptr + i, (INT8 *)serialWeight);
                ptr[i].vec = serialBias;
                ptr[i].mdt = DT_F32;
            } else if (ptr[i].mdt == DT_I4) {
                // trans w from 4 to 32
                TransWeightFromI4ToF32(ptr + i, (INT8 *)serialWeight);
                ptr[i].vec = serialBias;
                ptr[i].mdt = DT_F32;
            } else if (ptr[i].mdt == DT_BIN01 || ptr[i].mdt == DT_BIN11) {
                TransWeightFromBINToF32(ptr + i, (INT8 *)serialWeight);
                TransBiasFromF16ToF32(ptr + i, (F16 *)serialBias);
                ptr[i].mdt = DT_F32;
            } else {
                ptr[i].weight = serialWeight;
                ptr[i].vec = serialBias;
            }
        }

        if (spec->dt == DT_F16) {
            if (ptr[i].mdt == DT_F32) {
                // trans w&b from 32 to 16
                TransWeightFromF32ToF16(ptr + i, (F32 *)serialWeight);
                TransBiasFromF32ToF16(ptr + i, (F32 *)serialBias);
                ptr[i].mdt = DT_F16;
            } else if ((ptr[i].mdt == DT_I8) || (ptr[i].mdt == DT_F16_8Q)) {
                // trans w from 8 to 16
                TransWeightFromI8ToF16(ptr + i, (INT8 *)serialWeight);
                ptr[i].vec = serialBias;
                ptr[i].mdt = DT_F16;
            } else if (ptr[i].mdt == DT_I4) {
                // trans w from 4 to 16
                TransWeightFromI4ToF16(ptr + i, (INT8 *)serialWeight);
                ptr[i].vec = serialBias;
                ptr[i].mdt = DT_F16;
            } else {
                ptr[i].weight = serialWeight;
                ptr[i].vec = serialBias;
            }
        }

        if (spec->dt == DT_F32_8Q) {
            if (ptr[i].mdt == DT_F32_8Q) {
                // trans w from 8 to 32
                TransWeightFromI8ToF32(ptr + i, (INT8 *)serialWeight);
                ptr[i].vec = serialBias;
                ptr[i].mdt = DT_F32;
            } else if (ptr[i].mdt == DT_F16) {
                // trans w&b from 16 to 32
                TransWeightFromF16ToF32(ptr + i, (F16 *)serialWeight);
                TransBiasFromF16ToF32(ptr + i, (F16 *)serialBias);
                ptr[i].mdt = DT_F32;
            } else if (ptr[i].mdt == DT_I4) {
                // trans w from 4 to 8
                if (sharedWeightDataTypeMap.count(ptr[i].op_name) && IS_X86_AVX512(arch)) {
                    ptr[i].mdt = DT_U8_Q;
                    TransWeightFromI4ToU8(ptr + i, (INT8 *)serialWeight);
                } else {
                    ptr[i].mdt = DT_I8;
                    TransWeightFromI4ToI8(ptr + i, (INT8 *)serialWeight);
                }
                ptr[i].vec = serialBias;
            } else {
                ptr[i].weight = serialWeight;
                ptr[i].vec = serialBias;
            }
        }

        if (spec->dt == DT_F16_8Q) {
            if (ptr[i].mdt == DT_F32) {
                // trans w&b from 32 to 16
                TransWeightFromF32ToF16(ptr + i, (F32 *)serialWeight);
                TransBiasFromF32ToF16(ptr + i, (F32 *)serialBias);
                ptr[i].mdt = DT_F16;
            } else if (ptr[i].mdt == DT_F32_8Q) {
                // trans w from 8 to 16
                // trans b from 32 to 16
                TransWeightFromI8ToF16(ptr + i, (INT8 *)serialWeight);
                TransBiasFromF32ToF16(ptr + i, (F32 *)serialBias);
                ptr[i].mdt = DT_F16;
            } else if (ptr[i].mdt == DT_I8) {
                // trans b from 32 to 16
                ptr[i].weight = serialWeight;
                TransBiasFromF32ToF16(ptr + i, (F32 *)serialBias);
            } else if (ptr[i].mdt == DT_I4) {
                // trans w from 4 to 8
                // trans b from 32 to 16
                TransWeightFromI4ToI8(ptr + i, (INT8 *)serialWeight);
                TransBiasFromF32ToF16(ptr + i, (F32 *)serialBias);
                ptr[i].mdt = DT_I8;
            } else {
                ptr[i].weight = serialWeight;
                ptr[i].vec = serialBias;
            }
        }
        sharedWeightDataTypeMap[ptr[i].op_name] = ptr[i].mdt;
    }

    for (int i = 0; i < spec->num_operator_specs; i++) {
        if (OT_SharedWeight == spec->ops[i].type) {
            std::set<DataType> innerTypes = {DT_F32_8Q, DT_F16_8Q, DT_I8, DT_I4, DT_F16, DT_F32};
            DataType &sdt = spec->ops[i].ps.shared_weight_spec.desc.dt;
            if (innerTypes.count(sdt)) {
                sdt = sharedWeightDataTypeMap[spec->ops[i].name];
                if (IS_X86_AVX512(arch) && (sdt == DT_I8)) {
                    sdt = DT_U8_Q;
                }
            }
        }
        if (OT_PreAllocatedMemory == spec->ops[i].type) {
            DataType &pdt = spec->ops[i].ps.preallocated_memory_spec.desc.dt;
            if ((pdt == DT_F16) || (pdt == DT_F32)) {
                pdt = noQuantDataType(spec->dt);
            }
        }
    }
    return SUCCESS;
}

EE deserialize_model(const U8 *bytes, ModelSpec *spec, DataType targetDt)
{
    U32 pos = 0;
    EE ret = deserialize_header(bytes, spec, targetDt, &pos);
    if (ret == SUCCESS) {
        ret = deserialize_operator(bytes, spec, &pos);
    }
    if (ret == SUCCESS) {
        ret = deserialize_weight(bytes, spec, &pos);
    }
#ifdef _USE_OP_TENSOR_RELATIONS
    if (ret == SUCCESS) {
        ret = operator_relationship(spec);
    }
#endif
    if (spec->file->stream_mode) {
        spec->file->length = pos;
    }
    spec->version = sg_boltVersion;
    return ret;
}

static void *func(void *arg)
{
    ModelFileDescriptor *file = (ModelFileDescriptor *)arg;
    volatile char sum = 0;
    for (U32 i = 0; i < file->length; i += 4 * 1024) {
        sum = file->content[i];
    }
    return NULL;
}

EE deserialize_model_from_file(
    const char *filePath, ModelSpec *spec, DataType targetDt, bool useFileStream)
{
    UNI_DEBUG_LOG("Read bolt model from %s...\n", (useFileStream ? "file stream" : filePath));
    EE ret = NOT_SUPPORTED;
    UNI_PROFILE(
        {
            spec->file = (ModelFileDescriptor *)mt_malloc(sizeof(ModelFileDescriptor));
            UNI_MEMSET(spec->file, 0, sizeof(ModelFileDescriptor));
            spec->file->stream_mode = useFileStream;
            if (spec->file->stream_mode) {
                spec->file->content = (U8 *)filePath;
                spec->file->length = INT_MAX;
                ret = SUCCESS;
            } else {
#if defined(__GLIBC__) || defined(__linux__)
                spec->file->file = open(filePath, O_RDONLY);
                if (-1 == spec->file->file) {
                    UNI_ERROR_LOG("Cannot open bolt model file %s.\n", filePath);
                    return FILE_ERROR;
                }
                struct stat ss;
                if (-1 == fstat(spec->file->file, &ss)) {
                    UNI_ERROR_LOG("Cannot get size from bolt model file %s descriptor.\n", filePath);
                    return FILE_ERROR;
                }
                spec->file->length = ss.st_size;
                spec->file->content = (U8 *)mmap(
                    nullptr, spec->file->length, PROT_READ, MAP_SHARED, spec->file->file, 0);
                if (MAP_FAILED == spec->file->content) {
                    UNI_ERROR_LOG("Map bolt model file %s failed.\n", filePath);
                    return FILE_ERROR;
                }
#elif defined(_WIN32)
                spec->file->file = CreateFile(
                    filePath, GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, NULL);
                if (spec->file->file == INVALID_HANDLE_VALUE) {
                    UNI_ERROR_LOG("Cannot open bolt model file %s.\n", filePath);
                    return FILE_ERROR;
                }
                LARGE_INTEGER length;
                if (!GetFileSizeEx(spec->file->file, &length)) {
                    UNI_ERROR_LOG("Cannot get size from bolt model file %s descriptor.\n", filePath);
                    return FILE_ERROR;
                }
                spec->file->length = length.QuadPart;
                spec->file->map = CreateFileMapping(spec->file->file, NULL, PAGE_READONLY, 0, 0, 0);
                if (spec->file->map == NULL) {
                    UNI_ERROR_LOG("CreateFileMapping failed\n");
                    return FILE_ERROR;
                }
                spec->file->content = (U8 *)MapViewOfFile(spec->file->map, FILE_MAP_READ, 0, 0, 0);
                if (spec->file->content == NULL) {
                    UNI_ERROR_LOG("Map bolt model file %s failed.\n", filePath);
                    return FILE_ERROR;
                }
                int ret = pthread_create(&(spec->file->thread), NULL, func, spec->file);
                if (ret != 0) {
                    UNI_ERROR_LOG("pthread create failed.\n");
                }
#else
                EE ret =
                    load_binary(filePath, (void **)&(spec->file->content), &(spec->file->length));
                if (ret != SUCCESS) {
                    return ret;
                }
#endif
            }
            ret = deserialize_model(spec->file->content, spec, targetDt);
        },
        std::string("deserialize_model_from_file"), std::string("prepare"));
    UNI_DEBUG_LOG("Read bolt model end.\n");
    return ret;
}
