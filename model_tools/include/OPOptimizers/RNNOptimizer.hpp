// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_RNNOPTIMIZER
#define _H_RNNOPTIMIZER

#include "OPOptimizer.hpp"
#include <stack>
#include <set>

class RNNOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        hasOptimized |= simplifyONNXRNN(spec);

        char *environmentSetting = getenv("BOLT_PADDING");
        bool usePadding =
            (environmentSetting != NULL && std::string(environmentSetting) == std::string("ON"))
            ? true
            : false;
        if (usePadding) {
            hasOptimized |= padding(spec);
        }
        return hasOptimized;
    }

    bool padding(ModelSpec *spec)
    {
        int align = 32;
        bool hasOptimized = false;
        OperatorType constantOfShape[1] = {OT_ConstantOfShape};
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_RNN) {
                RNNParamSpec param = spec->ops[i].ps.rnn_spec;
                if (param.numOutput % align == 0) {
                    continue;
                }
                // currently not support to padding PLSTM
                if (param.numProjection > 0 && param.numProjection % align != 0) {
                    continue;
                }
                UNI_INFO_LOG("padding RNN/GRU/LSTM operator %s's hidden states to 32 times.\n",
                    spec->ops[i].name);
                int weightIndex = searchWeightIndex(spec, spec->ops[i].name);
                CHECK_REQUIREMENT(weightIndex >= 0);
                int directions;
                if (param.biDirection) {
                    directions = 2;
                } else {
                    directions = 1;
                }
                int gates = 0;
                switch (param.mode) {
                    case RNN_LSTM:
                        gates = 4;
                        break;
                    case RNN_GRU:
                    case RNN_GRU_LBR:
                        gates = 3;
                        break;
                    default:
                        continue;
                }
                int paddingOutput = (param.numOutput + align - 1) / align * align;
                U32 oldNum =
                    directions * gates * param.numOutput * bytesOf(spec->ws[weightIndex].mdt);
                U32 newNum = directions * gates * paddingOutput * bytesOf(spec->ws[weightIndex].mdt);
                int inputDim = spec->ws[weightIndex].bytes_of_weight / oldNum;
                int paddingInputDim = inputDim - param.numOutput + paddingOutput;
                U32 oldVecNum = oldNum;
                U32 newVecNum = newNum;
                if (param.mode == RNN_GRU_LBR) {
                    oldVecNum += directions * param.numOutput * bytesOf(spec->ws[weightIndex].mdt);
                    newVecNum += directions * paddingOutput * bytesOf(spec->ws[weightIndex].mdt);
                }

                CHECK_REQUIREMENT(spec->ws[weightIndex].bytes_of_weight == oldNum * inputDim);
                CHECK_REQUIREMENT(spec->ws[weightIndex].bytes_of_vec == 0 ||
                    spec->ws[weightIndex].bytes_of_vec == oldVecNum);
                U8 *oldWeight = spec->ws[weightIndex].weight;
                U8 *oldVec = spec->ws[weightIndex].vec;

                spec->ops[i].ps.rnn_spec.numOutput = paddingOutput;
                spec->ws[weightIndex].bytes_of_weight = newNum * paddingInputDim;
                spec->ws[weightIndex].weight =
                    (U8 *)mt_new_storage(spec->ws[weightIndex].bytes_of_weight);
                spec->ws[weightIndex].bytes_of_vec =
                    (spec->ws[weightIndex].bytes_of_vec > 0) ? newVecNum : 0;
                spec->ws[weightIndex].vec = (U8 *)mt_new_storage(spec->ws[weightIndex].bytes_of_vec);
                U8 *newWeight = spec->ws[weightIndex].weight;
                U8 *newVec = spec->ws[weightIndex].vec;

                int oldVecStride = param.numOutput * bytesOf(spec->ws[weightIndex].mdt);
                int oldWeightStride = oldVecStride * inputDim;
                int newVecStride = paddingOutput * bytesOf(spec->ws[weightIndex].mdt);
                int newWeightStride = newVecStride * paddingInputDim;
                int vecBlank =
                    (paddingOutput - param.numOutput) * bytesOf(spec->ws[weightIndex].mdt);
                int weightBlank = vecBlank * paddingInputDim;
                for (int j = 0, wid = 0, vid = 0; j < directions; j++) {
                    for (int k = 0; k < gates; k++, wid++, vid++) {
                        for (U32 m = 0; m < param.numOutput; m++) {
                            int id00 = (wid * param.numOutput + m) * inputDim *
                                bytesOf(spec->ws[weightIndex].mdt);
                            int id01 = (wid * paddingOutput + m) * paddingInputDim *
                                bytesOf(spec->ws[weightIndex].mdt);
                            int copySize = inputDim * bytesOf(spec->ws[weightIndex].mdt);
                            memcpy(newWeight + id01, oldWeight + id00, copySize);
                            memset(newWeight + id01 + copySize, 0,
                                (paddingInputDim - inputDim) * bytesOf(spec->ws[weightIndex].mdt));
                        }
                        int id01 = (wid * paddingOutput + param.numOutput) * paddingInputDim *
                            bytesOf(spec->ws[weightIndex].mdt);
                        memset(newWeight + id01, 0, weightBlank);
                        if (oldVec != nullptr) {
                            int id10 = vid * oldVecStride;
                            int id11 = vid * newVecStride;
                            memcpy(newVec + id11, oldVec + id10, oldVecStride);
                            memset(newVec + id11 + oldVecStride, 0, vecBlank);
                        }
                    }
                    if (param.mode == RNN_GRU_LBR && oldVec != nullptr) {
                        int id10 = vid * oldVecStride;
                        int id11 = vid * newVecStride;
                        memcpy(newVec + id11, oldVec + id10, oldVecStride);
                        memset(newVec + id11 + oldVecStride, 0, vecBlank);
                        vid++;
                    }
                }
                delete oldWeight;
                if (oldVec != nullptr) {
                    delete oldVec;
                }

                std::string name = std::string(spec->ops[i].name) + std::string("_slice");
                OperatorSpec tfsliceOperator = mt_create_operator(name.c_str(), OT_TfSlice, 1, 1);
                TfSliceParamSpec tfSlicePs;
                memset(&tfSlicePs, 0, sizeof(tfSlicePs));
                tfSlicePs.dim_size = 3;
                for (U32 j = 0; j < tfSlicePs.dim_size; j++) {
                    tfSlicePs.begin[j] = 0;
                    tfSlicePs.end[j] = -1;
                    tfSlicePs.strides[j] = 1;
                    tfSlicePs.begin_mask[j] = 1;
                    tfSlicePs.end_mask[j] = 1;
                }
                tfSlicePs.end[2] = param.numOutput;
                tfSlicePs.end_mask[2] = 0;
                tfsliceOperator.ps.tfslice_spec = tfSlicePs;
                str_copy(tfsliceOperator.output_tensors_name[0],
                    spec->ops[i].output_tensors_name[0], NAME_LEN);
                str_copy(tfsliceOperator.input_tensors_name[0], name.c_str(), name.length());
                str_copy(spec->ops[i].output_tensors_name[0], name.c_str(), name.length());
                mt_insert_operator(spec, i + 1, tfsliceOperator);
                hasOptimized = true;
                i++;
            }
        }
        return hasOptimized;
    }

    bool simplifyONNXRNN(ModelSpec *spec)
    {
        bool hasOptimized = false;
        OperatorType constantOfShape[1] = {OT_ConstantOfShape};
        int lastConstantOfShapeId = -1;
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_RNN && spec->ops[i].num_inputs >= 1 &&
                spec->ops[i].ps.rnn_spec.steps >= 0) {
                // remove [1, 0, 2] transpose before RNN
                std::vector<std::pair<int, int>> prevOpIndexes =
                    searchOperatorIndexByOutput(spec, spec->ops[i].input_tensors_name[0], 0, i);
                int rnnInputId = i;
                if (prevOpIndexes.size() == 1) {
                    int prevOpIndex = prevOpIndexes[0].first;
                    if (spec->ops[prevOpIndex].type == OT_Transpose &&
                        spec->ops[prevOpIndex].ps.transpose_spec.trans_size == 3 &&
                        spec->ops[prevOpIndex].ps.transpose_spec.trans_dims[0] == 1 &&
                        spec->ops[prevOpIndex].ps.transpose_spec.trans_dims[1] == 0 &&
                        spec->ops[prevOpIndex].ps.transpose_spec.trans_dims[2] == 2) {
                        std::vector<std::pair<int, int>> nextOpIndexes = searchOperatorIndexByInput(
                            spec, spec->ops[prevOpIndex].output_tensors_name[0], prevOpIndex,
                            spec->num_operator_specs);
                        // 1 -> N
                        if (nextOpIndexes.size() != 1) {
                            UNI_ERROR_LOG("RNNOptimizer can not process Transpose before RNN, "
                                          "1->N\n");
                        }
                        str_copy(spec->ops[i].input_tensors_name[0],
                            spec->ops[prevOpIndex].input_tensors_name[0], NAME_LEN);
                        setOperatorInvalid(spec, prevOpIndex);
                        std::vector<std::pair<int, int>> prevOpIndexes1 =
                            searchOperatorIndexByOutput(
                                spec, spec->ops[prevOpIndex].input_tensors_name[0], 0, prevOpIndex);
                        if (prevOpIndexes1.size() > 0) {
                            rnnInputId = prevOpIndexes1[0].first;
                        }
                    }
                }
                int constantOfShapeId;
                if (rnnInputId == i) {
                    constantOfShapeId = lastConstantOfShapeId;
                } else {
                    lastConstantOfShapeId = constantOfShapeId =
                        searchOperatorIndexBackward(spec, i, constantOfShape, 1, false, rnnInputId);
                }

                std::stack<int> recursive;
                std::set<int> duplicate;
                recursive.push(i);
                duplicate.insert(i);
                while (recursive.size() > 0 && constantOfShapeId != -1) {
                    int index = recursive.top();
                    UNI_DEBUG_LOG("process node %d name %s num_inputs %d\n", index,
                        spec->ops[index].name, spec->ops[index].num_inputs);
                    recursive.pop();
                    U32 inputNum = spec->ops[index].num_inputs;
                    // if RNN node, skip first input
                    if (index == i) {
                        inputNum--;
                    }
                    for (U32 j = 0; j < inputNum; j++) {
                        int inputId = j;
                        // if RNN node, skip first input
                        if (index == i) {
                            inputId++;
                        }
                        std::string name = spec->ops[index].input_tensors_name[inputId];
                        UNI_DEBUG_LOG(
                            "search node %d input %d name %s\n", index, inputId, name.c_str());
                        std::vector<std::pair<int, int>> prevOpIndexes =
                            searchOperatorIndexByOutput(spec, name, 0, index);
                        // model input
                        if (name != std::string("") && prevOpIndexes.size() == 0) {
                            int id = -1;
                            for (int k = 0; k < spec->num_inputs; k++) {
                                if (spec->input_names[k] == name) {
                                    id = k;
                                    break;
                                }
                            }
                            if (id == -1) {
                                UNI_ERROR_LOG(
                                    "RNNOptimizer encounter unknown tensor %s\n", name.c_str());
                                //} else {
                                //    // erase model input
                                //    for (int k = id + 1; k < spec->num_outputs; k++) {
                                //        str_copy(
                                //            spec->input_names[k - 1], spec->input_names[k], NAME_LEN);
                                //    }
                                //    delete spec->input_names[spec->num_inputs - 1];
                                //    spec->num_inputs--;
                            }
                        }
                        // invalid model
                        if (prevOpIndexes.size() > 1) {
                            UNI_ERROR_LOG("RNNOptimizer encounter multi-operators have save output "
                                          "name %s\n",
                                name.c_str());
                        }
                        // recursive process
                        if (prevOpIndexes.size() == 1) {
                            int prevOpIndex = prevOpIndexes[0].first;
                            std::vector<std::pair<int, int>> nextOpIndexes =
                                searchOperatorIndexByInput(
                                    spec, name, prevOpIndex + 1, spec->num_operator_specs);
                            UNI_DEBUG_LOG("previous operator %d's output %s -> %d operators\n",
                                prevOpIndex, name.c_str(), (int)nextOpIndexes.size());
                            bool same = true;
                            for (U32 k = 1; k < nextOpIndexes.size(); k++) {
                                if (nextOpIndexes[k].first != nextOpIndexes[k - 1].first) {
                                    same = false;
                                    break;
                                }
                            }
                            // 1 -> N
                            if (!same) {
                                continue;
                            } else {
                                if (duplicate.find(prevOpIndex) == duplicate.end()) {
                                    recursive.push(prevOpIndex);
                                    duplicate.insert(prevOpIndex);
                                }
                            }
                        }
                    }
                    if (inputNum == spec->ops[index].num_inputs) {
                        setOperatorInvalid(spec, index);
                    }
                    if (index == i) {
                        for (U32 k = 1; k < spec->ops[index].num_inputs; k++) {
                            delete spec->ops[index].input_tensors_name[k];
                        }
                        spec->ops[index].num_inputs = 1;
                        for (U32 k = 1; k < spec->ops[index].num_outputs; k++) {
                            delete spec->ops[index].output_tensors_name[k];
                        }
                        spec->ops[index].num_outputs = 1;
                    }
                    hasOptimized = true;
                }

                // node after rnn
                std::vector<std::pair<int, int>> nextOpIndexes1 = searchOperatorIndexByInput(
                    spec, spec->ops[i].output_tensors_name[0], i + 1, spec->num_operator_specs);
                if (nextOpIndexes1.size() == 1) {
                    int nextOpIndex1 = nextOpIndexes1[0].first;
                    std::vector<std::pair<int, int>> nextOpIndexes2 = searchOperatorIndexByInput(
                        spec, spec->ops[nextOpIndex1].output_tensors_name[0], nextOpIndex1 + 1,
                        spec->num_operator_specs);
                    if (nextOpIndexes2.size() != 1) {
                        continue;
                    }
                    int nextOpIndex2 = nextOpIndexes2[0].first;

                    // onnx-scan + transpose(1,0,2)
                    if (spec->ops[nextOpIndex1].type == OT_Transpose &&
                        spec->ops[nextOpIndex1].ps.transpose_spec.trans_size == 3 &&
                        spec->ops[nextOpIndex1].ps.transpose_spec.trans_dims[0] == 1 &&
                        spec->ops[nextOpIndex1].ps.transpose_spec.trans_dims[1] == 0 &&
                        spec->ops[nextOpIndex1].ps.transpose_spec.trans_dims[2] == 2) {
                        str_copy(spec->ops[i].output_tensors_name[0],
                            spec->ops[nextOpIndex1].output_tensors_name[0], NAME_LEN);
                        setOperatorInvalid(spec, nextOpIndex1);
                    }
                    // onnx-rnn + reshape(-1,0,0)/squeeze(axis=1) + transpose(1,0,2)/rnn
                    if (spec->ops[nextOpIndex1].type == OT_Squeeze ||
                        spec->ops[nextOpIndex1].type == OT_Reshape) {
                        str_copy(spec->ops[i].output_tensors_name[0],
                            spec->ops[nextOpIndex1].output_tensors_name[0], NAME_LEN);
                        setOperatorInvalid(spec, nextOpIndex1);
                        if (spec->ops[nextOpIndex2].type == OT_Transpose &&
                            spec->ops[nextOpIndex2].ps.transpose_spec.trans_size == 3 &&
                            spec->ops[nextOpIndex2].ps.transpose_spec.trans_dims[0] == 1 &&
                            spec->ops[nextOpIndex2].ps.transpose_spec.trans_dims[1] == 0 &&
                            spec->ops[nextOpIndex2].ps.transpose_spec.trans_dims[2] == 2) {
                            str_copy(spec->ops[i].output_tensors_name[0],
                                spec->ops[nextOpIndex2].output_tensors_name[0], NAME_LEN);
                            setOperatorInvalid(spec, nextOpIndex2);
                        }
                    }
                    // onnx-rnn + transpose(2,0,1,3) + reshape(0,0,-1)
                    if (spec->ops[nextOpIndex1].type == OT_Transpose &&
                        spec->ops[nextOpIndex1].ps.transpose_spec.trans_size == 4 &&
                        spec->ops[nextOpIndex1].ps.transpose_spec.trans_dims[0] == 2 &&
                        spec->ops[nextOpIndex1].ps.transpose_spec.trans_dims[1] == 0 &&
                        spec->ops[nextOpIndex1].ps.transpose_spec.trans_dims[2] == 1 &&
                        spec->ops[nextOpIndex1].ps.transpose_spec.trans_dims[3] == 3 &&
                        spec->ops[nextOpIndex2].type == OT_Reshape) {
                        str_copy(spec->ops[i].output_tensors_name[0],
                            spec->ops[nextOpIndex2].output_tensors_name[0], NAME_LEN);
                        setOperatorInvalid(spec, nextOpIndex1);
                        setOperatorInvalid(spec, nextOpIndex2);
                    }
                    // onnx-birnn + transpose(0,2,1,3) + reshape(0,0,-1) + transpose(1,0,2)/rnn
                    if (spec->ops[nextOpIndex1].type == OT_Transpose &&
                        spec->ops[nextOpIndex1].ps.transpose_spec.trans_size == 4 &&
                        spec->ops[nextOpIndex1].ps.transpose_spec.trans_dims[0] == 0 &&
                        spec->ops[nextOpIndex1].ps.transpose_spec.trans_dims[1] == 2 &&
                        spec->ops[nextOpIndex1].ps.transpose_spec.trans_dims[2] == 1 &&
                        spec->ops[nextOpIndex1].ps.transpose_spec.trans_dims[3] == 3 &&
                        spec->ops[nextOpIndex2].type == OT_Reshape) {
                        str_copy(spec->ops[i].output_tensors_name[0],
                            spec->ops[nextOpIndex2].output_tensors_name[0], NAME_LEN);
                        setOperatorInvalid(spec, nextOpIndex1);
                        setOperatorInvalid(spec, nextOpIndex2);

                        std::vector<std::pair<int, int>> nextOpIndexes3 = searchOperatorIndexByInput(
                            spec, spec->ops[nextOpIndex2].output_tensors_name[0], nextOpIndex2 + 1,
                            spec->num_operator_specs);
                        if (nextOpIndexes3.size() != 1) {
                            continue;
                        }
                        int nextOpIndex3 = nextOpIndexes3[0].first;
                        if (spec->ops[nextOpIndex3].type == OT_Transpose &&
                            spec->ops[nextOpIndex3].ps.transpose_spec.trans_size == 3 &&
                            spec->ops[nextOpIndex3].ps.transpose_spec.trans_dims[0] == 1 &&
                            spec->ops[nextOpIndex3].ps.transpose_spec.trans_dims[1] == 0 &&
                            spec->ops[nextOpIndex3].ps.transpose_spec.trans_dims[2] == 2) {
                            str_copy(spec->ops[i].output_tensors_name[0],
                                spec->ops[nextOpIndex3].output_tensors_name[0], NAME_LEN);
                            setOperatorInvalid(spec, nextOpIndex3);
                        }
                    }
                }
            }
        }
        return hasOptimized;
    }
};
#endif
