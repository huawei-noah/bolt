// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MultiHeadAttentionOPTIMIZER
#define _H_MultiHeadAttentionOPTIMIZER

#include <map>
#include "OPOptimizer.hpp"

class MultiHeadAttentionOptimizer : public OPOptimizer {
    int upstreamOpIndex(ModelSpec *spec, int curIndex, std::string curInputName)
    {
        for (int i = curIndex - 1; i >= 0; i--) {
            if (spec->ops[i].type == OT_None) {
                continue;
            }
            if (spec->ops[i].num_outputs == 1) {
                std::string curOutputName = spec->ops[i].output_tensors_name[0];
                if (curInputName.compare(curOutputName) == 0) {
                    return i;
                }
            }
        }
        return -1;
    }

    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;

        std::map<OperatorType, int> firLoopMap;
        std::map<OperatorType, int> secLoopMap;
        int firEltIndex = -1;
        int secEltIndex = -1;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            bool firstBoolTag = false;
            bool secBoolTag = false;

            int backIpIndex1 = -1;
            int backIpIndex2 = -1;
            int backLayerNorm1Index = -1;
            ActivationMode globalActi = ACTIVATION_RELU;

            int backIpIndex = -1;
            int backReshapeIndex = -1;
            int leftReshapeIndex = -1;
            int midReshapeIndex = -1;
            int rightReshapeIndex = -1;
            int leftIpIndex = -1;
            int midIpIndex = -1;
            int rightIpIndex = -1;
            int globalPowerIndex = -1;

            if (spec->ops[i].type == OT_LayerNorm) {
                int layerNormOpIndex = i;
                std::string lnOutput = spec->ops[layerNormOpIndex].output_tensors_name[0];
                for (int j = i + 1; j < spec->num_operator_specs; j++) {
                    if (spec->ops[j].type == OT_None) {
                        continue;
                    }
                    if (secEltIndex != -1) {
                        break;
                    }
                    if (spec->ops[j].type == OT_Eltwise && firEltIndex == -1) {
                        firEltIndex = j;
                        continue;
                    }

                    if (spec->ops[j].type == OT_Eltwise && firEltIndex != -1) {
                        secEltIndex = j;
                        continue;
                    }

                    if (firEltIndex == -1) {
                        if (firLoopMap.find(spec->ops[j].type) == firLoopMap.end()) {
                            firLoopMap.insert(std::pair<OperatorType, int>(spec->ops[j].type, 1));
                        } else {
                            firLoopMap[spec->ops[j].type] = firLoopMap[spec->ops[j].type] + 1;
                        }
                    } else {
                        if (secLoopMap.find(spec->ops[j].type) == secLoopMap.end()) {
                            secLoopMap.insert(std::pair<OperatorType, int>(spec->ops[j].type, 1));
                        } else {
                            secLoopMap[spec->ops[j].type] = secLoopMap[spec->ops[j].type] + 1;
                        }
                    }
                }

                int secLoopMap_size = 0;
                std::map<OperatorType, int>::iterator secLoop_iter;
                secLoop_iter = secLoopMap.begin();
                while (secLoop_iter != secLoopMap.end()) {
                    secLoopMap_size += secLoop_iter->second;
                    secLoop_iter++;
                }
                if (secLoopMap_size == 4) {
                    std::string secEltInput0 = spec->ops[secEltIndex].input_tensors_name[0];
                    std::string secEltInput1 = spec->ops[secEltIndex].input_tensors_name[1];

                    std::string secLoopEltPreFcName = "";
                    std::string secLoopEltPreNotFcName = "";
                    int secLoopEltPreIndex0 = upstreamOpIndex(spec, secEltIndex, secEltInput0);
                    if (spec->ops[secLoopEltPreIndex0].type == OT_FC) {
                        secLoopEltPreFcName = secEltInput0;
                        secLoopEltPreNotFcName = secEltInput1;
                    } else {
                        secLoopEltPreFcName = secEltInput1;
                        secLoopEltPreNotFcName = secEltInput0;
                    }

                    backIpIndex1 = upstreamOpIndex(spec, secEltIndex, secLoopEltPreFcName);
                    if (!(backIpIndex1 != -1 && spec->ops[backIpIndex1].type == OT_FC)) {
                        continue;
                    }

                    std::string backIpIndex1Input = spec->ops[backIpIndex1].input_tensors_name[0];
                    int actiIndex = upstreamOpIndex(spec, backIpIndex1, backIpIndex1Input);
                    if (actiIndex != -1) {
                        if (spec->ops[actiIndex].type == OT_Relu) {
                            globalActi = ACTIVATION_RELU;
                        } else if (spec->ops[actiIndex].type == OT_Gelu) {
                            globalActi = ACTIVATION_GELU;
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    }

                    std::string actiInput = spec->ops[actiIndex].input_tensors_name[0];
                    backIpIndex2 = upstreamOpIndex(spec, actiIndex, actiInput);
                    if (!(backIpIndex2 != -1 && spec->ops[backIpIndex2].type == OT_FC)) {
                        continue;
                    }

                    std::string backIpIndex2Input = spec->ops[backIpIndex2].input_tensors_name[0];
                    backLayerNorm1Index = upstreamOpIndex(spec, backIpIndex2, backIpIndex2Input);
                    if (!(backLayerNorm1Index != -1 &&
                            spec->ops[backLayerNorm1Index].type == OT_LayerNorm)) {
                        continue;
                    }

                    std::string firstEltOutput = spec->ops[firEltIndex].output_tensors_name[0];
                    std::string backLayerNorm1Output =
                        spec->ops[backLayerNorm1Index].output_tensors_name[0];
                    if (secLoopEltPreNotFcName.compare(firstEltOutput) == 0) {
                        secBoolTag = false;
                    } else if (secLoopEltPreNotFcName.compare(backLayerNorm1Output) == 0) {
                        secBoolTag = true;
                    } else {
                        continue;
                    }

                } else {
                    continue;
                }

                int firLoopMap_size = 0;
                std::map<OperatorType, int>::iterator firLoop_iter;
                firLoop_iter = firLoopMap.begin();
                while (firLoop_iter != firLoopMap.end()) {
                    firLoopMap_size += firLoop_iter->second;
                    firLoop_iter++;
                }
                if (firLoopMap_size == 16) {
                    std::string firstEltInput0 = spec->ops[firEltIndex].input_tensors_name[0];
                    std::string firstEltInput1 = spec->ops[firEltIndex].input_tensors_name[1];

                    std::string firLoopEltPreFcName = "";
                    std::string firLoopEltPreNotFcName = "";
                    int firLoopEltPreIndex0 = upstreamOpIndex(spec, firEltIndex, firstEltInput0);
                    if (spec->ops[firLoopEltPreIndex0].type == OT_FC) {
                        firLoopEltPreFcName = firstEltInput0;
                        firLoopEltPreNotFcName = firstEltInput1;
                    } else {
                        firLoopEltPreFcName = firstEltInput1;
                        firLoopEltPreNotFcName = firstEltInput0;
                    }

                    backIpIndex = upstreamOpIndex(spec, firEltIndex, firLoopEltPreFcName);
                    if (!(backIpIndex != -1 && spec->ops[backIpIndex].type == OT_FC)) {
                        continue;
                    }

                    std::string backIpInput = spec->ops[backIpIndex].input_tensors_name[0];
                    backReshapeIndex = upstreamOpIndex(spec, backIpIndex, backIpInput);
                    if (!(backReshapeIndex != -1 && spec->ops[backReshapeIndex].type == OT_Reshape)) {
                        continue;
                    }

                    std::string backReshapeInput = spec->ops[backReshapeIndex].input_tensors_name[0];
                    int backTransposeIndex =
                        upstreamOpIndex(spec, backReshapeIndex, backReshapeInput);
                    if (!(backTransposeIndex != -1 &&
                            spec->ops[backTransposeIndex].type == OT_Transpose)) {
                        continue;
                    }

                    std::string backTransposeInput =
                        spec->ops[backTransposeIndex].input_tensors_name[0];
                    int backMatmulIndex1 =
                        upstreamOpIndex(spec, backTransposeIndex, backTransposeInput);
                    if (!(backMatmulIndex1 != -1 && spec->ops[backMatmulIndex1].type == OT_MatMul)) {
                        continue;
                    }

                    std::string backMatmulIndex1Input0 =
                        spec->ops[backMatmulIndex1].input_tensors_name[0];
                    std::string backMatmulIndex1Input1 =
                        spec->ops[backMatmulIndex1].input_tensors_name[1];
                    int leftTransposeIndex =
                        upstreamOpIndex(spec, backMatmulIndex1, backMatmulIndex1Input1);
                    if (!(leftTransposeIndex != -1 &&
                            spec->ops[leftTransposeIndex].type == OT_Transpose)) {
                        continue;
                    }

                    std::string leftTransposeInput =
                        spec->ops[leftTransposeIndex].input_tensors_name[0];
                    leftReshapeIndex = upstreamOpIndex(spec, leftTransposeIndex, leftTransposeInput);
                    if (!(leftReshapeIndex != -1 && spec->ops[leftReshapeIndex].type == OT_Reshape)) {
                        continue;
                    }

                    std::string leftReshapeInput = spec->ops[leftReshapeIndex].input_tensors_name[0];
                    leftIpIndex = upstreamOpIndex(spec, leftReshapeIndex, leftReshapeInput);
                    if (!(leftIpIndex != -1 && spec->ops[leftIpIndex].type == OT_FC)) {
                        continue;
                    }

                    int softmaxIndex =
                        upstreamOpIndex(spec, backMatmulIndex1, backMatmulIndex1Input0);
                    if (!(softmaxIndex != -1 && spec->ops[softmaxIndex].type == OT_Softmax)) {
                        continue;
                    }

                    std::string softmaxInput = spec->ops[softmaxIndex].input_tensors_name[0];
                    int preSoftmaxIndex = upstreamOpIndex(spec, softmaxIndex, softmaxInput);
                    int backMatmulIndex2 = -1;
                    if (preSoftmaxIndex != -1 &&
                        (spec->ops[preSoftmaxIndex].type == OT_Scale ||
                            spec->ops[preSoftmaxIndex].type == OT_Power)) {
                        globalPowerIndex = preSoftmaxIndex;
                        std::string scaleInput = spec->ops[preSoftmaxIndex].input_tensors_name[0];
                        backMatmulIndex2 = upstreamOpIndex(spec, preSoftmaxIndex, scaleInput);
                    } else if (preSoftmaxIndex != -1 &&
                        spec->ops[preSoftmaxIndex].type == OT_MatMul) {
                        backMatmulIndex2 = preSoftmaxIndex;
                    } else {
                        continue;
                    }
                    if (!(backMatmulIndex2 != -1 && spec->ops[backMatmulIndex2].type == OT_MatMul)) {
                        continue;
                    }

                    std::string backMatmulIndex2Input0 =
                        spec->ops[backMatmulIndex2].input_tensors_name[0];
                    int midBackOpIndex =
                        upstreamOpIndex(spec, backMatmulIndex2, backMatmulIndex2Input0);
                    int midTransposeIndex = -1;
                    if (midBackOpIndex != -1 &&
                        (spec->ops[midBackOpIndex].type == OT_Scale ||
                            spec->ops[midBackOpIndex].type == OT_Power)) {
                        globalPowerIndex = midBackOpIndex;
                        std::string scaleInput = spec->ops[midBackOpIndex].input_tensors_name[0];
                        midTransposeIndex = upstreamOpIndex(spec, midBackOpIndex, scaleInput);
                    } else if (midBackOpIndex != -1 &&
                        spec->ops[midBackOpIndex].type == OT_Transpose) {
                        midTransposeIndex = midBackOpIndex;
                    }
                    if (!(midTransposeIndex != -1 &&
                            spec->ops[midTransposeIndex].type == OT_Transpose)) {
                        continue;
                    }

                    std::string midTransposeInput =
                        spec->ops[midTransposeIndex].input_tensors_name[0];
                    midReshapeIndex = upstreamOpIndex(spec, midTransposeIndex, midTransposeInput);
                    if (!(midReshapeIndex != -1 && spec->ops[midReshapeIndex].type == OT_Reshape)) {
                        continue;
                    }

                    std::string midReshapeInput = spec->ops[midReshapeIndex].input_tensors_name[0];
                    midIpIndex = upstreamOpIndex(spec, midReshapeIndex, midReshapeInput);
                    if (!(midIpIndex != -1 && spec->ops[midIpIndex].type == OT_FC)) {
                        continue;
                    }

                    std::string backMatmulIndex2Input1 =
                        spec->ops[backMatmulIndex2].input_tensors_name[1];
                    int rightBackOpIndex =
                        upstreamOpIndex(spec, backMatmulIndex2, backMatmulIndex2Input1);
                    int rightTransposeIndex = -1;
                    if (rightBackOpIndex != -1 &&
                        (spec->ops[rightBackOpIndex].type == OT_Scale ||
                            spec->ops[rightBackOpIndex].type == OT_Power)) {
                        globalPowerIndex = rightBackOpIndex;
                        std::string scaleInput = spec->ops[rightBackOpIndex].input_tensors_name[0];
                        rightTransposeIndex = upstreamOpIndex(spec, rightBackOpIndex, scaleInput);
                    } else if (rightBackOpIndex != -1 &&
                        spec->ops[rightBackOpIndex].type == OT_Transpose) {
                        rightTransposeIndex = rightBackOpIndex;
                    }
                    if (!(rightTransposeIndex != -1 &&
                            spec->ops[rightTransposeIndex].type == OT_Transpose)) {
                        continue;
                    }

                    std::string rightTransposeInput =
                        spec->ops[rightTransposeIndex].input_tensors_name[0];
                    rightReshapeIndex =
                        upstreamOpIndex(spec, rightTransposeIndex, rightTransposeInput);
                    if (!(midReshapeIndex != -1 && spec->ops[midReshapeIndex].type == OT_Reshape)) {
                        continue;
                    }

                    std::string rightReshapeInput =
                        spec->ops[rightReshapeIndex].input_tensors_name[0];
                    rightIpIndex = upstreamOpIndex(spec, rightReshapeIndex, rightReshapeInput);
                    if (!(rightIpIndex != -1 && spec->ops[rightIpIndex].type == OT_FC)) {
                        continue;
                    }

                    std::string ln2ip0 = spec->ops[leftIpIndex].input_tensors_name[0];
                    std::string ln2ip1 = spec->ops[midIpIndex].input_tensors_name[0];
                    std::string ln2ip2 = spec->ops[rightIpIndex].input_tensors_name[0];
                    if (ln2ip0.compare(ln2ip1) && ln2ip1.compare(ln2ip2)) {
                        if (firLoopEltPreNotFcName.compare(ln2ip0) == 0) {
                            firstBoolTag = true;
                        } else {
                            secBoolTag = false;
                        }
                    }

                    MultiheadAttentionParamSpec multihead_spec;
                    multihead_spec.fc_desc[0] = spec->ops[leftIpIndex].ps.fc_spec;
                    multihead_spec.fc_desc[1] = spec->ops[midIpIndex].ps.fc_spec;
                    multihead_spec.fc_desc[2] = spec->ops[rightIpIndex].ps.fc_spec;
                    multihead_spec.fc_desc[3] = spec->ops[backIpIndex].ps.fc_spec;
                    multihead_spec.fc_desc[4] = spec->ops[backIpIndex2].ps.fc_spec;
                    multihead_spec.fc_desc[5] = spec->ops[backIpIndex1].ps.fc_spec;
                    multihead_spec.power_spec = spec->ops[globalPowerIndex].ps.power_spec;
                    multihead_spec.eltwiseWithLayerNormIn[0] = firstBoolTag;
                    multihead_spec.eltwiseWithLayerNormIn[1] = secBoolTag;
                    multihead_spec.actiMode = globalActi;
                    multihead_spec.reshapeDesc[0] = spec->ops[leftReshapeIndex].ps.reshape_spec;
                    multihead_spec.reshapeDesc[1] = spec->ops[midReshapeIndex].ps.reshape_spec;
                    multihead_spec.reshapeDesc[2] = spec->ops[rightReshapeIndex].ps.reshape_spec;
                    multihead_spec.reshapeDesc[3] = spec->ops[backReshapeIndex].ps.reshape_spec;
                    multihead_spec.eltwiseDesc[0] = spec->ops[firEltIndex].ps.eltwise_spec;
                    multihead_spec.eltwiseDesc[1] = spec->ops[secEltIndex].ps.eltwise_spec;

                    spec->ops[layerNormOpIndex].type = OT_MultiHeadAttention;
                    spec->ops[layerNormOpIndex].ps.multiheadAttention_spec = multihead_spec;

                    int lnWeightIndex = searchWeightIndex(spec, spec->ops[layerNormOpIndex].name);
                    int leftIpWeightIndex = searchWeightIndex(spec, spec->ops[leftIpIndex].name);
                    int midIpWeightIndex = searchWeightIndex(spec, spec->ops[midIpIndex].name);
                    int rightIpWeightIndex = searchWeightIndex(spec, spec->ops[rightIpIndex].name);
                    int backIpWeightIndex = searchWeightIndex(spec, spec->ops[backIpIndex].name);
                    int backLayerNorm1WeightIndex =
                        searchWeightIndex(spec, spec->ops[backLayerNorm1Index].name);
                    int backIpWeightIndex2 = searchWeightIndex(spec, spec->ops[backIpIndex2].name);
                    int backIpWeightIndex1 = searchWeightIndex(spec, spec->ops[backIpIndex1].name);

                    U32 weightSize = spec->ws[lnWeightIndex].bytes_of_weight +
                        spec->ws[leftIpWeightIndex].bytes_of_weight +
                        spec->ws[midIpWeightIndex].bytes_of_weight +
                        spec->ws[rightIpWeightIndex].bytes_of_weight +
                        spec->ws[backIpWeightIndex].bytes_of_weight +
                        spec->ws[backLayerNorm1WeightIndex].bytes_of_weight +
                        spec->ws[backIpWeightIndex2].bytes_of_weight +
                        spec->ws[backIpWeightIndex1].bytes_of_weight;
                    U32 biasSize = spec->ws[lnWeightIndex].bytes_of_vec +
                        spec->ws[leftIpWeightIndex].bytes_of_vec +
                        spec->ws[midIpWeightIndex].bytes_of_vec +
                        spec->ws[rightIpWeightIndex].bytes_of_vec +
                        spec->ws[backIpWeightIndex].bytes_of_vec +
                        spec->ws[backLayerNorm1WeightIndex].bytes_of_vec +
                        spec->ws[backIpWeightIndex2].bytes_of_vec +
                        spec->ws[backIpWeightIndex1].bytes_of_vec;
                    U8 *multihead_weight = (U8 *)mt_new_storage(weightSize);
                    U8 *multihead_vec = (U8 *)mt_new_storage(biasSize);
                    int weightOffset = 0;
                    memcpy(&multihead_weight[weightOffset], spec->ws[lnWeightIndex].weight,
                        spec->ws[lnWeightIndex].bytes_of_weight);
                    weightOffset += spec->ws[lnWeightIndex].bytes_of_weight;
                    memcpy(&multihead_weight[weightOffset], spec->ws[leftIpWeightIndex].weight,
                        spec->ws[leftIpWeightIndex].bytes_of_weight);
                    weightOffset += spec->ws[leftIpWeightIndex].bytes_of_weight;
                    memcpy(&multihead_weight[weightOffset], spec->ws[midIpWeightIndex].weight,
                        spec->ws[midIpWeightIndex].bytes_of_weight);
                    weightOffset += spec->ws[midIpWeightIndex].bytes_of_weight;
                    memcpy(&multihead_weight[weightOffset], spec->ws[rightIpWeightIndex].weight,
                        spec->ws[rightIpWeightIndex].bytes_of_weight);
                    weightOffset += spec->ws[rightIpWeightIndex].bytes_of_weight;
                    memcpy(&multihead_weight[weightOffset], spec->ws[backIpWeightIndex].weight,
                        spec->ws[backIpWeightIndex].bytes_of_weight);
                    weightOffset += spec->ws[backIpWeightIndex].bytes_of_weight;
                    memcpy(&multihead_weight[weightOffset],
                        spec->ws[backLayerNorm1WeightIndex].weight,
                        spec->ws[backLayerNorm1WeightIndex].bytes_of_weight);
                    weightOffset += spec->ws[backLayerNorm1WeightIndex].bytes_of_weight;
                    memcpy(&multihead_weight[weightOffset], spec->ws[backIpWeightIndex2].weight,
                        spec->ws[backIpWeightIndex2].bytes_of_weight);
                    weightOffset += spec->ws[backIpWeightIndex2].bytes_of_weight;
                    memcpy(&multihead_weight[weightOffset], spec->ws[backIpWeightIndex1].weight,
                        spec->ws[backIpWeightIndex1].bytes_of_weight);

                    int vecOffset = 0;
                    memcpy(&multihead_vec[vecOffset], spec->ws[lnWeightIndex].vec,
                        spec->ws[lnWeightIndex].bytes_of_vec);
                    vecOffset += spec->ws[lnWeightIndex].bytes_of_vec;
                    memcpy(&multihead_vec[vecOffset], spec->ws[leftIpWeightIndex].vec,
                        spec->ws[leftIpWeightIndex].bytes_of_vec);
                    vecOffset += spec->ws[leftIpWeightIndex].bytes_of_vec;
                    memcpy(&multihead_vec[vecOffset], spec->ws[midIpWeightIndex].vec,
                        spec->ws[midIpWeightIndex].bytes_of_vec);
                    vecOffset += spec->ws[midIpWeightIndex].bytes_of_vec;
                    memcpy(&multihead_vec[vecOffset], spec->ws[rightIpWeightIndex].vec,
                        spec->ws[rightIpWeightIndex].bytes_of_vec);
                    vecOffset += spec->ws[rightIpWeightIndex].bytes_of_vec;
                    memcpy(&multihead_vec[vecOffset], spec->ws[backIpWeightIndex].vec,
                        spec->ws[backIpWeightIndex].bytes_of_vec);
                    vecOffset += spec->ws[backIpWeightIndex].bytes_of_vec;
                    memcpy(&multihead_vec[vecOffset], spec->ws[backLayerNorm1WeightIndex].vec,
                        spec->ws[backLayerNorm1WeightIndex].bytes_of_vec);
                    vecOffset += spec->ws[backLayerNorm1WeightIndex].bytes_of_vec;
                    memcpy(&multihead_vec[vecOffset], spec->ws[backIpWeightIndex2].vec,
                        spec->ws[backIpWeightIndex2].bytes_of_vec);
                    vecOffset += spec->ws[backIpWeightIndex2].bytes_of_vec;
                    memcpy(&multihead_vec[vecOffset], spec->ws[backIpWeightIndex1].vec,
                        spec->ws[backIpWeightIndex1].bytes_of_vec);
                    spec->ws[lnWeightIndex].bytes_of_weight = weightSize;
                    spec->ws[lnWeightIndex].bytes_of_vec = biasSize;

                    if (outOfFileMapRange(spec->ws[lnWeightIndex].weight, spec->mfd)) {
                        delete spec->ws[lnWeightIndex].weight;
                    }
                    if (outOfFileMapRange(spec->ws[lnWeightIndex].vec, spec->mfd)) {
                        delete spec->ws[lnWeightIndex].vec;
                    }
                    spec->ws[lnWeightIndex].weight = multihead_weight;
                    spec->ws[lnWeightIndex].vec = multihead_vec;

                    memcpy(spec->ops[layerNormOpIndex].output_tensors_name[0],
                        spec->ops[secEltIndex].output_tensors_name[0], NAME_LEN);

                    for (int k = layerNormOpIndex + 1; k <= secEltIndex; k++) {
                        setOperatorInvalid(spec, k);
                    }

                    for (int k = lnWeightIndex + 1; k <= backIpWeightIndex1; k++) {
                        setWeightOperatorInvalid(spec, k);
                    }

                    hasOptimized = true;

                    firEltIndex = -1;
                    secEltIndex = -1;
                    firLoopMap.clear();
                    secLoopMap.clear();
                }
            }
        }
        return hasOptimized;
    }
};
#endif
