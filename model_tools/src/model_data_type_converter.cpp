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
#include <math.h>
#include "model_data_type_converter.h"
#include "OPOptimizers/DeprecatedOPOptimizer.hpp"

// Return the weight scale
F32 ws_datatype_converter_bnn(U8 *originalPtr, U8 *targetPtr, int paramNum)
{
    F32 *f32PtrParam = (F32 *)originalPtr;
    BIN8 *targetPtrParam = (BIN8 *)targetPtr;
    for (int i = 0; i < paramNum; i += 8) {
        BIN8 temp = 0;  // Initialize all bits to 0
        for (int j = 0; j < 8; j++) {
            U32 bitNo = 7 - j;
            if (f32PtrParam[i + j] >
                0) {  // Set bit if weight is positive. Works for both DOREFA and XNOR
                temp |= (1 << bitNo);
            }
        }
        targetPtrParam[i / 8] = temp;
    }

    F32 scale = 1;
    for (int i = 0; i < paramNum; i++) {
        scale = f32PtrParam[i];
        if (scale > 0) {
            break;
        }
    }
    return scale;
}

// return quantization scale
F32 ws_datatype_converter_int8(U8 *originalPtr, U8 *targetPtr, int paramNum)
{
    F32 *f32PtrParam = (F32 *)originalPtr;
    INT8 *targetPtrParam = (INT8 *)targetPtr;

    F32 maxabs = 0;
    for (int i = 0; i < paramNum; i++) {
        if (abs(f32PtrParam[i]) > maxabs) {
            maxabs = abs(f32PtrParam[i]);
        }
    }

    F32 scale = 127.0 / maxabs;
    for (int i = 0; i < paramNum; i++) {
        targetPtrParam[i] = round(f32PtrParam[i] * scale);
    }
    return scale;
}

inline EE getTargetDataType(DataConvertType convertMode, DataType *type)
{
    if (*type != DT_F32) {
        return SUCCESS;
    }

    switch (convertMode) {
        case F32_to_F32: {
            *type = DT_F32;
            break;
        }
        case F32_to_F16: {
            *type = DT_F16;
            break;
        }
        case F32_to_I8: {
            *type = DT_I8;
            break;
        }
        default:
            return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline DataType get_storage_type(
    ModelSpec *ms, std::string opName, std::string storageMode, DataType originalType)
{
    if ("NOQUANT" == storageMode) {
        return originalType;
    }
    if ("FP16" == storageMode) {
        return DT_F16;
    }
    for (int i = 0; i < ms->num_operator_specs; i++) {
        std::string name = ms->ops[i].name;
        if (name == opName) {
            auto opType = ms->ops[i].type;
            if (OT_LayerNorm == opType || OT_Scale == opType || OT_PRelu == opType) {
                return originalType;
            }
            if ("INT8" == storageMode) {
                return DT_I8;
            }
            if (1 == ms->ops[i].num_quant_feature && 1 == ms->ops[i].feature_scale[0].num_scale &&
                0 == ms->ops[i].feature_scale[0].scale[0]) {
                return originalType;
            } else {
                return DT_I8;
            }
        }
    }
    UNI_ERROR_LOG("No OP found with name %s\n", opName.c_str());
    return originalType;
}

EE ms_datatype_converter(
    ModelSpec *originalMs, ModelSpec *targetMs, DataConvertType convertMode, std::string storageMode)
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
    targetMs->input_dims = (TensorDesc *)mt_new_storage(targetMs->num_inputs * sizeof(TensorDesc));
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
    OperatorSpec *opsPtr =
        (OperatorSpec *)mt_new_storage(targetMs->num_operator_specs * sizeof(OperatorSpec));
    std::map<std::string, DataType> weightDataTypeMap;

    for (int i = 0; i < targetMs->num_operator_specs; i++) {
        str_copy(opsPtr[i].name, originalMs->ops[i].name, NAME_LEN);
        opsPtr[i].type = originalMs->ops[i].type;
        opsPtr[i].num_inputs = originalMs->ops[i].num_inputs;
        opsPtr[i].input_tensors_name = (I8 **)mt_new_storage(opsPtr[i].num_inputs * sizeof(I8 *));
        for (U32 j = 0; j < opsPtr[i].num_inputs; j++) {
            opsPtr[i].input_tensors_name[j] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
            memcpy(opsPtr[i].input_tensors_name[j], originalMs->ops[i].input_tensors_name[j],
                NAME_LEN);
        }
        opsPtr[i].num_outputs = originalMs->ops[i].num_outputs;
        opsPtr[i].output_tensors_name = (I8 **)mt_new_storage(opsPtr[i].num_outputs * sizeof(I8 *));
        for (U32 j = 0; j < opsPtr[i].num_outputs; j++) {
            opsPtr[i].output_tensors_name[j] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
            memcpy(opsPtr[i].output_tensors_name[j], originalMs->ops[i].output_tensors_name[j],
                NAME_LEN);
        }

        if (OT_None != opsPtr[i].type) {
            U32 numTensors = opsPtr[i].num_inputs + opsPtr[i].num_outputs;
            opsPtr[i].tensor_positions = (I32 *)mt_new_storage(numTensors * sizeof(I32));
            memcpy(opsPtr[i].tensor_positions, originalMs->ops[i].tensor_positions,
                numTensors * sizeof(I32));
        } else {
            opsPtr[i].tensor_positions = nullptr;
        }

        opsPtr[i].num_quant_feature = originalMs->ops[i].num_quant_feature;
        if (0 == opsPtr[i].num_quant_feature) {
            opsPtr[i].feature_scale = nullptr;
        } else {
            opsPtr[i].feature_scale =
                (QuantSpec *)mt_new_storage(opsPtr[i].num_quant_feature * sizeof(QuantSpec));
            for (U32 j = 0; j < opsPtr[i].num_quant_feature; j++) {
                opsPtr[i].feature_scale[j].num_scale = originalMs->ops[i].feature_scale[j].num_scale;
                int num = opsPtr[i].feature_scale[j].num_scale;

                opsPtr[i].feature_scale[j].scale = (F32 *)mt_new_storage(num * sizeof(F32));
                memcpy(opsPtr[i].feature_scale[j].scale, originalMs->ops[i].feature_scale[j].scale,
                    num * sizeof(F32));
            }
        }

        opsPtr[i].ps = originalMs->ops[i].ps;

        switch (opsPtr[i].type) {
            case OT_SharedWeight: {
                weightDataTypeMap[opsPtr[i].name] = opsPtr[i].ps.shared_weight_spec.desc.dt;
                CHECK_STATUS(
                    getTargetDataType(convertMode, &(opsPtr[i].ps.shared_weight_spec.desc.dt)));
                break;
            }
            case OT_PreAllocatedMemory: {
                CHECK_STATUS(getTargetDataType(
                    convertMode, &(opsPtr[i].ps.preallocated_memory_spec.desc.dt)));
                break;
            }
            default:
                break;
        }
    }
    targetMs->ops = opsPtr;
    targetMs->num_weight_specs = originalMs->num_weight_specs;
    WeightSpec *wsPtr =
        (WeightSpec *)mt_new_storage(targetMs->num_weight_specs * sizeof(WeightSpec));
    for (int i = 0; i < targetMs->num_weight_specs; i++) {
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
            if (wdt == DT_F32 || wdt == DT_F16) {
                wsPtr[i].mdt = get_storage_type(targetMs, wsPtr[i].op_name, storageMode, wdt);
            }

            weightNum = originalMs->ws[i].bytes_of_weight / bytesOf(originalMs->ws[i].mdt);
            wsPtr[i].bytes_of_weight = weightNum * bytesOf(wsPtr[i].mdt);
        }

        wsPtr[i].num_quant_scale = originalMs->ws[i].num_quant_scale;
        if (0 == wsPtr[i].num_quant_scale) {
            wsPtr[i].weight_scale = nullptr;
        } else {
            wsPtr[i].weight_scale =
                (QuantSpec *)mt_new_storage(wsPtr[i].num_quant_scale * sizeof(QuantSpec));
            for (U32 j = 0; j < wsPtr[i].num_quant_scale; j++) {
                wsPtr[i].weight_scale[j].num_scale = originalMs->ws[i].weight_scale[j].num_scale;
                int num = wsPtr[i].weight_scale[j].num_scale;

                wsPtr[i].weight_scale[j].scale = (F32 *)mt_new_storage(num * sizeof(F32));
                memcpy(wsPtr[i].weight_scale[j].scale, originalMs->ws[i].weight_scale[j].scale,
                    num * sizeof(F32));
            }
        }

        wsPtr[i].weight = (U8 *)mt_new_storage(wsPtr[i].bytes_of_weight);

        DataType vdt = DT_F32;
        int biasNum = originalMs->ws[i].bytes_of_vec / bytesOf(DT_F32);
        CHECK_STATUS(getTargetDataType(convertMode, &vdt));
        if (DT_F32 == vdt && DT_F16 == wsPtr[i].mdt) {
            vdt = DT_F16;
        }
        wsPtr[i].bytes_of_vec = biasNum * bytesOf(vdt);
        wsPtr[i].vec = (U8 *)mt_new_storage(wsPtr[i].bytes_of_vec);

        if (DT_I32 == originalMs->ws[i].mdt || DT_U32 == originalMs->ws[i].mdt) {
            if (wsPtr[i].bytes_of_weight > 0) {
                memcpy(wsPtr[i].weight, originalMs->ws[i].weight, wsPtr[i].bytes_of_weight);
            }
            if (wsPtr[i].bytes_of_vec > 0) {
                memcpy(wsPtr[i].vec, originalMs->ws[i].vec, wsPtr[i].bytes_of_vec);
            }
        } else {
            switch (wsPtr[i].mdt) {
                case DT_I32:
                case DT_U32:
                case DT_F32:
                case DT_F16: {
                    transformFromFloat(wsPtr[i].mdt, (float *)originalMs->ws[i].weight,
                        wsPtr[i].weight, weightNum);
                    transformFromFloat(
                        wsPtr[i].mdt, (float *)originalMs->ws[i].vec, wsPtr[i].vec, biasNum);
                    break;
                }
                case DT_I8: {
                    F32 scale = ws_datatype_converter_int8(
                        originalMs->ws[i].weight, wsPtr[i].weight, weightNum);
                    wsPtr[i].num_quant_scale = 1;
                    wsPtr[i].weight_scale = (QuantSpec *)mt_new_storage(sizeof(QuantSpec));
                    wsPtr[i].weight_scale[0].num_scale = 1;
                    wsPtr[i].weight_scale[0].scale = (F32 *)mt_new_storage(sizeof(F32));
                    wsPtr[i].weight_scale[0].scale[0] = scale;

                    transformFromFloat(vdt, (float *)originalMs->ws[i].vec, wsPtr[i].vec, biasNum);
                    break;
                }
                case DT_BIN01:
                case DT_BIN11: {
                    F32 scale = ws_datatype_converter_bnn(
                        originalMs->ws[i].weight, wsPtr[i].weight, weightNum);
                    transformFromFloat(
                        DT_F16, (float *)originalMs->ws[i].vec, wsPtr[i].vec, biasNum);
                    // Fuse the weight scale
                    if (1 != scale) {
                        F32 value;
                        for (int i = 0; i < biasNum / 2; i++) {
                            transformToFloat(DT_F16, wsPtr[i].vec + i * bytesOf(DT_F16), &value, 1);
                            value *= scale;
                            transformFromFloat(
                                DT_F16, &value, wsPtr[i].vec + i * bytesOf(DT_F16), 1);
                        }
                    }
                    break;
                }
                default:
                    return NOT_SUPPORTED;
            }
        }
    }
    targetMs->ws = wsPtr;

    if (nullptr != originalMs->op_relationship_entries) {
        targetMs->num_op_tensor_entries = originalMs->num_op_tensor_entries;
        targetMs->op_relationship_entries = (OperatorRelationshipMapEntry *)mt_new_storage(
            targetMs->num_op_tensor_entries * sizeof(OperatorRelationshipMapEntry));
        for (int i = 0; i < targetMs->num_op_tensor_entries; i++) {
            str_copy(targetMs->op_relationship_entries[i].op,
                originalMs->op_relationship_entries[i].op, NAME_LEN);

            targetMs->op_relationship_entries[i].num_inputs =
                originalMs->op_relationship_entries[i].num_inputs;
            targetMs->op_relationship_entries[i].input_op_names = (I8 **)mt_new_storage(
                targetMs->op_relationship_entries[i].num_inputs * sizeof(I8 *));
            for (U32 j = 0; j < targetMs->op_relationship_entries[i].num_inputs; j++) {
                targetMs->op_relationship_entries[i].input_op_names[j] =
                    (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                str_copy(targetMs->op_relationship_entries[i].input_op_names[j],
                    originalMs->op_relationship_entries[i].input_op_names[j], NAME_LEN);
            }

            targetMs->op_relationship_entries[i].num_outputs =
                originalMs->op_relationship_entries[i].num_outputs;
            targetMs->op_relationship_entries[i].output_op_names = (I8 **)mt_new_storage(
                targetMs->op_relationship_entries[i].num_outputs * sizeof(I8 *));
            for (U32 j = 0; j < targetMs->op_relationship_entries[i].num_outputs; j++) {
                targetMs->op_relationship_entries[i].output_op_names[j] =
                    (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                str_copy(targetMs->op_relationship_entries[i].output_op_names[j],
                    originalMs->op_relationship_entries[i].output_op_names[j], NAME_LEN);
            }
        }
    } else {
        targetMs->num_op_tensor_entries = 0;
        targetMs->op_relationship_entries = nullptr;
    }
    return SUCCESS;
}
