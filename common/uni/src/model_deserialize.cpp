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
#include <fstream>
#include <map>
#include <set>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "model_serialize_deserialize.hpp"
#include "profiling.h"

int get_operator_parameter_size(OperatorType operatorType)
{
    std::map<OperatorType, int> operatorParameterSizeMap = {{OT_Conv, sizeof(ConvolutionParamSpec)},
        {OT_Deconvolution, sizeof(ConvolutionParamSpec)}, {OT_FC, sizeof(FullyConnectedParamSpec)},
        {OT_RNN, sizeof(RNNParamSpec)}, {OT_MatMul, sizeof(MatMulParamSpec)},
        {OT_Resize, sizeof(ResizeParamSpec)},
        {OT_BilateralSliceApply, sizeof(BilateralSliceApplyParamSpec)},
        {OT_Pooling, sizeof(PoolingParamSpec)}, {OT_Scale, sizeof(ScaleParamSpec)},
        {OT_BatchNorm, sizeof(BatchNormParamSpec)}, {OT_Reduction, sizeof(ReductionParamSpec)},
        {OT_ArgMax, sizeof(ArgMaxParamSpec)}, {OT_Softmax, sizeof(SoftmaxParamSpec)},
        {OT_Clip, sizeof(ClipParamSpec)}, {OT_Power, sizeof(PowerParamSpec)},
        {OT_Relu, sizeof(ReLUParamSpec)}, {OT_Gather, sizeof(GatherParamSpec)},
        {OT_Embedding, sizeof(EmbedParamSpec)}, {OT_Pad, sizeof(PadParamSpec)},
        {OT_Eltwise, sizeof(EltwiseParamSpec)}, {OT_Concat, sizeof(ConcatParamSpec)},
        {OT_Slice, sizeof(SliceParamSpec)}, {OT_TfSlice, sizeof(TfSliceParamSpec)},
        {OT_Cast, sizeof(CastParamSpec)}, {OT_Transpose, sizeof(TransposeParamSpec)},
        {OT_Reshape, sizeof(ReshapeParamSpec)}, {OT_Squeeze, sizeof(SqueezeParamSpec)},
        {OT_Unsqueeze, sizeof(UnsqueezeParamSpec)}, {OT_Space2Depth, sizeof(Space2DepthParamSpec)},
        {OT_Depth2Space, sizeof(Depth2SpaceParamSpec)},
        {OT_ChannelResize, sizeof(ChannelResizeParamSpec)},
        {OT_PreAllocatedMemory, sizeof(PreAllocatedMemoryParamSpec)},
        {OT_SharedWeight, sizeof(SharedWeightParamSpec)}, {OT_Copy, sizeof(CopyParamSpec)},
        {OT_Check, sizeof(CheckParamSpec)}, {OT_Repeat, sizeof(RepeatParamSpec)},
        {OT_Attention, sizeof(AttentionParamSpec)},
        {OT_AttentionMask, sizeof(AttentionMaskParamSpec)},
        {OT_RelativePositionEmbedding, sizeof(EmbedParamSpec)},
        {OT_RelativeShift, sizeof(RelativeShiftParamSpec)}, {OT_PriorBox, sizeof(PriorBoxParamSpec)},
        {OT_DetectionOutput, sizeof(DetectionOutputParamSpec)},
        {OT_Yolov3DetectionOutput, sizeof(Yolov3DetectionOutputParamSpec)},
        {OT_MultiHeadAttention, sizeof(MultiheadAttentionParamSpec)},
        {OT_Tile, sizeof(TileParamSpec)}, {OT_Splice, sizeof(SpliceParamSpec)}};
    int size;
    if (operatorParameterSizeMap.find(operatorType) == operatorParameterSizeMap.end()) {
        size = 0;
    } else {
        size = operatorParameterSizeMap[operatorType];
    }
    return size;
}

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

template <typename T>
void dequantize_int8_weight(int num, F32 scale, INT8 *q, T *d)
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

inline void dequantize_fp16(int num, unsigned short *q, F32 *d)
{
#if defined(_USE_NEON) && defined(__aarch64__)
    F16 *half = (F16 *)q;
#else
    U32 *word = (U32 *)d;
#endif

    for (int i = 0; i < num; i++) {
#if defined(_USE_NEON) && defined(__aarch64__)
        d[i] = half[i];
#else
        unsigned short value = q[i];
        unsigned short sign = (value & 0x8000) >> 15;
        unsigned short exponent = (value & 0x7c00) >> 10;
        unsigned short significand = value & 0x03FF;

        U32 u;
        if (exponent == 0) {
            if (significand == 0) {
                u = sign << 31;
            } else {
                exponent = 0;
                while (0 == (significand & 0x200)) {
                    significand <<= 1;
                    exponent++;
                }
                significand <<= 1;
                significand &= 0x3FF;
                u = (sign << 31) | ((-exponent + (-15 + 127)) << 23) | (significand << 13);
            }
        } else if (exponent == 0x1F) {
            u = (sign << 31) | (0xFF << 23) | (significand << 13);
        } else {
            u = (sign << 31) | ((exponent + (-15 + 127)) << 23) | (significand << 13);
        }
        word[i] = u;
#endif
    }
}

EE deserialize_header(const char *bytes, ModelSpec *spec, U32 *pos)
{
    const char *pointer = bytes + *pos;
    memcpy(&spec->version, pointer, sizeof(I32));
    pointer += sizeof(I32);
    *pos += sizeof(I32);
    if (spec->version != sg_boltVersion) {
        UNI_ERROR_LOG("X2bolt version is [%d], but your model version is : [%d].\n Please update "
                      "X2bolt to version[%d].\n",
            sg_boltVersion, spec->version, spec->version);
        CHECK_STATUS(NOT_MATCH);
        return NOT_MATCH;
    }

    memcpy(&spec->magic_number, pointer, sizeof(I32));
    pointer += sizeof(I32);
    *pos += sizeof(I32);
    if (spec->magic_number != sg_magicNumber) {
        UNI_ERROR_LOG(
            "magic_number not_match: code %d bolt model %d\n", sg_magicNumber, spec->magic_number);
        CHECK_STATUS(NOT_MATCH);
        return NOT_MATCH;
    }

    str_copy(spec->model_name, pointer, NAME_LEN);
    pointer += NAME_LEN;
    *pos += NAME_LEN;

    spec->dt = *((DataType *)pointer);
    pointer += sizeof(DataType);
    *pos += sizeof(DataType);

    spec->num_inputs = *((I32 *)pointer);
    pointer += sizeof(I32);
    *pos += sizeof(I32);

    spec->input_names = (I8 **)mt_new_storage(spec->num_inputs * sizeof(I8 *));
    for (int i = 0; i < spec->num_inputs; i++) {
        spec->input_names[i] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
        str_copy(spec->input_names[i], pointer, NAME_LEN);
        pointer += NAME_LEN;
        *pos += NAME_LEN;
    }

    spec->input_dims = (TensorDesc *)mt_new_storage(spec->num_inputs * sizeof(TensorDesc));
    memcpy(spec->input_dims, pointer, spec->num_inputs * sizeof(TensorDesc));
    pointer += spec->num_inputs * sizeof(TensorDesc);
    *pos += spec->num_inputs * sizeof(TensorDesc);

    spec->num_outputs = *((I32 *)pointer);
    pointer += sizeof(I32);
    *pos += sizeof(I32);

    spec->output_names = (I8 **)mt_new_storage(spec->num_outputs * NAME_LEN);
    for (int i = 0; i < spec->num_outputs; i++) {
        spec->output_names[i] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
        str_copy(spec->output_names[i], pointer, NAME_LEN);
        pointer += NAME_LEN;
        *pos += NAME_LEN;
    }
    return SUCCESS;
}

EE deserialize_operator(const char *bytes, ModelSpec *spec, U32 *pos)
{
    const char *pointer = bytes + *pos;
    I32 *p4numOperatorSpecs = (I32 *)pointer;
    spec->num_operator_specs = *p4numOperatorSpecs;
    pointer += sizeof(U32);
    *pos += sizeof(U32);

    OperatorSpec *ptr =
        (OperatorSpec *)mt_new_storage(spec->num_operator_specs * sizeof(OperatorSpec));
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
        for (U32 j = 0; j < ptr[i].num_inputs; j++) {
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
        ptr[i].tensor_positions = (I32 *)mt_new_storage(numTensors * sizeof(I32));
        memcpy(ptr[i].tensor_positions, pointer, numTensors * sizeof(I32));
        pointer += numTensors * sizeof(I32);
        *pos += numTensors * sizeof(I32);

        ptr[i].num_quant_feature = *((U32 *)pointer);
        pointer += sizeof(U32);
        *pos += sizeof(U32);

        if (0 != ptr[i].num_quant_feature) {
            ptr[i].feature_scale =
                (QuantSpec *)mt_new_storage(ptr[i].num_quant_feature * sizeof(QuantSpec));
        } else {
            ptr[i].feature_scale = nullptr;
        }
        for (U32 j = 0; j < ptr[i].num_quant_feature; j++) {
            ptr[i].feature_scale[j].num_scale = *((int *)pointer);
            int num = ptr[i].feature_scale[j].num_scale;
            pointer += sizeof(int);
            *pos += sizeof(int);

            ptr[i].feature_scale[j].scale = (F32 *)mt_new_storage(num * sizeof(F32));
            memcpy(ptr[i].feature_scale[j].scale, pointer, num * sizeof(F32));
            pointer += num * sizeof(F32);
            *pos += num * sizeof(F32);
        }

        int operatorParameterSize = get_operator_parameter_size(ptr[i].type);
        memcpy(&(ptr[i].ps), pointer, operatorParameterSize);
        pointer += operatorParameterSize;
        *pos += operatorParameterSize;
    }

    return SUCCESS;
}

EE deserialize_weight(const char *bytes, ModelSpec *spec, U32 *pos)
{
    const char *pointer = bytes + *pos;
    I32 *p4numWeightSpecs = (I32 *)pointer;
    spec->num_weight_specs = *p4numWeightSpecs;
    pointer += sizeof(U32);
    *pos += sizeof(U32);

    WeightSpec *ptr = (WeightSpec *)mt_new_storage(spec->num_weight_specs * sizeof(WeightSpec));
    spec->ws = ptr;
    for (int i = 0; i < spec->num_weight_specs; i++) {
        U32 *length = (U32 *)pointer;
        pointer += sizeof(U32);
        *pos += sizeof(U32);
        U32 weightBiasBytes = 0;

        str_copy(ptr[i].op_name, pointer, NAME_LEN);
        pointer += NAME_LEN;
        *pos += NAME_LEN;

        memcpy(&(ptr[i].mdt), pointer, sizeof(DataType));
        pointer += sizeof(U32);
        *pos += sizeof(U32);

        bool quantFP16 = false;
        bool quantInt8 = false;
        if (DT_F16 == ptr[i].mdt && DT_F32 == spec->dt) {
            ptr[i].mdt = DT_F32;
            quantFP16 = true;
        } else if (DT_I8 == ptr[i].mdt && DT_I8 != spec->dt) {
            ptr[i].mdt = (spec->dt == DT_F16_8Q) ? DT_F16 : spec->dt;
            quantInt8 = true;
        }

        memcpy(&(ptr[i].bytes_of_weight), pointer, sizeof(U32));
        U32 alignSize = ptr[i].bytes_of_weight;

        if (quantFP16) {
            ptr[i].bytes_of_weight *= 2;
        }
        if (quantInt8) {
            ptr[i].bytes_of_weight *= bytesOf(ptr[i].mdt);
        }
        pointer += sizeof(U32);
        *pos += sizeof(U32);

        ptr[i].weight = (U8 *)mt_new_storage(ptr[i].bytes_of_weight);
        U8 *serialWeight = (U8 *)pointer;

        pointer += alignSize;
        *pos += alignSize;
        weightBiasBytes += alignSize;

        memcpy(&(ptr[i].bytes_of_vec), pointer, sizeof(U32));
        pointer += sizeof(U32);
        *pos += sizeof(U32);

        alignSize = ptr[i].bytes_of_vec;
        if (quantFP16) {
            ptr[i].bytes_of_vec *= 2;
        }
        U8 *serialBias = nullptr;
        if (0 != ptr[i].bytes_of_vec) {
            serialBias = (U8 *)pointer;
            ptr[i].vec = (U8 *)mt_new_storage(ptr[i].bytes_of_vec);
        } else {
            ptr[i].vec = nullptr;
        }

        pointer += alignSize;
        *pos += alignSize;
        weightBiasBytes += alignSize;

        memcpy(&(ptr[i].num_quant_scale), pointer, sizeof(U32));
        pointer += sizeof(U32);
        *pos += sizeof(U32);

        if (0 != ptr[i].num_quant_scale) {
            ptr[i].weight_scale =
                (QuantSpec *)mt_new_storage(ptr[i].num_quant_scale * sizeof(QuantSpec));
        }
        for (U32 j = 0; j < ptr[i].num_quant_scale; j++) {
            ptr[i].weight_scale[j].num_scale = *((int *)pointer);
            int num = ptr[i].weight_scale[j].num_scale;
            pointer += sizeof(int);
            *pos += sizeof(int);

            ptr[i].weight_scale[j].scale = (F32 *)mt_new_storage(num * sizeof(F32));
            memcpy(ptr[i].weight_scale[j].scale, pointer, num * sizeof(F32));
            pointer += num * sizeof(F32);
            *pos += num * sizeof(F32);
        }

        CHECK_REQUIREMENT(*length == weightBiasBytes);

        if (quantFP16) {
            dequantize_fp16(
                ptr[i].bytes_of_weight / 4, (unsigned short *)serialWeight, (F32 *)ptr[i].weight);
            dequantize_fp16(
                ptr[i].bytes_of_vec / 4, (unsigned short *)serialBias, (F32 *)ptr[i].vec);
        } else {
            if (quantInt8) {
                CHECK_REQUIREMENT(
                    1 == ptr[i].num_quant_scale && 1 == ptr[i].weight_scale[0].num_scale);
                F32 scale = ptr[i].weight_scale[0].scale[0];
                if (DT_F32 == ptr[i].mdt) {
                    dequantize_int8_weight<F32>(ptr[i].bytes_of_weight / 4, scale,
                        (INT8 *)serialWeight, (F32 *)ptr[i].weight);
                } else {
#ifdef __aarch64__
                    dequantize_int8_weight<F16>(ptr[i].bytes_of_weight / 2, scale,
                        (INT8 *)serialWeight, (F16 *)ptr[i].weight);
#endif
                }
            } else {
                memcpy(ptr[i].weight, serialWeight, ptr[i].bytes_of_weight);
            }
            memcpy(ptr[i].vec, serialBias, ptr[i].bytes_of_vec);
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
    return SUCCESS;
}

EE deserialize_model_from_file(const char *fn, ModelSpec *spec, bool useFileStream)
{
    UNI_PROFILE(
        {
            char *bytes = nullptr;
            int fd;
            int fileLength;
            if (useFileStream) {
                bytes = (char *)fn;
            } else {
                fd = open(fn, O_RDONLY);
                if (-1 == fd) {
                    UNI_ERROR_LOG("Cannot open .bolt file. Name: %s\n", fn);
                    return FILE_ERROR;
                }

                struct stat ss;
                if (-1 == fstat(fd, &ss)) {
                    UNI_ERROR_LOG("Cannot get size from file descriptor. File Name: %s\n", fn);
                    return FILE_ERROR;
                }

                fileLength = ss.st_size;
                bytes = (char *)mmap(nullptr, fileLength, PROT_READ, MAP_SHARED, fd, 0);
                if (MAP_FAILED == bytes) {
                    UNI_ERROR_LOG("Mmap failed. File Name: %s\n", fn);
                    return FILE_ERROR;
                }
            }

            CHECK_STATUS(deserialize_model(bytes, spec));

            if (!useFileStream) {
                munmap(bytes, fileLength);
                if (-1 != fd) {
                    close(fd);
                }
            }
        },
        std::string("deserialize_model_from_file"), std::string("prepare"));
    return SUCCESS;
}
