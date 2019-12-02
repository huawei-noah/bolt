// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_MODELOPTIMIZER
#define _H_MODELOPTIMIZER

#include <vector>
#include <memory>
#include <math.h>
#include <string.h>
#include <map>
#include <string>
#include <assert.h>
#include <iostream>
#include "model_tools.h"
#include "model_serialize_deserialize.hpp"


class OPOptimizer {
    public:
        /**
         * @param spec
         */
        virtual bool optimize(ModelSpec* spec) = 0;
        int searchWeightIndex(ModelSpec* spec, char* op_name) {
            if (spec->num_weight_specs <= 0) {
                return -1;
            }

            std::map<std::string, int> weightsIndex;
            for (int i=0; i < spec->num_weight_specs; i++) {
                std::string key = spec->ws[i].op_name;    // attention, this is static structure attribute
                weightsIndex[key] = i;
            }
            
            std::string opNameStr = op_name;
            std::map<std::string, int>::iterator iter = weightsIndex.find(opNameStr);
            if(iter == weightsIndex.end())
                return -1;
            else
                return weightsIndex[opNameStr];
        }

        bool isValidOperator(ModelSpec* spec, int index){
            if (index >= spec->num_operator_specs) {
                return false;
            }
                
            if (spec->ops[index].type != OT_None) {
                return true;
            } else {
                return false;
            }
        }

        void setOperatorInvalid(ModelSpec* spec, int index) {
            if (index >= spec->num_operator_specs || index < 0) {
                return;
            }
            spec->ops[index].type = OT_None;
        }

        int searchOperatorIndexBackward(ModelSpec* spec, int end, OperatorType *queryOps, int queryNum, bool unskip=true) {
            for (int i = end; i >= 0; i--) {
                if (isValidOperator(spec, i)) {
                    for (int j=0; j<queryNum; j++) {
                        OperatorType opType = queryOps[j];
                        if (spec->ops[i].type == opType) {
                            return i;
                        }
                    }
                    if (unskip) {
                        return -1;
                    }
                }
            }
            return -1;
        }

        int searchOperatorIndexForward(ModelSpec* spec, int start, OperatorType *queryOps, int queryNum, bool unskip=true){
            for (int i = start; i < spec->num_operator_specs; i++) {
                if (isValidOperator(spec, i)) {
                    for (int j=0; j<queryNum; j++) {
                        OperatorType opType = queryOps[j];
                        if(spec->ops[i].type == opType) {
                            return i;
                        }
                    }
                    if (unskip) {
                        return -1;
                    }
                }
            }
            return -1;
        }
};


class ConvBNOptimizer: public OPOptimizer {
    virtual bool optimize(ModelSpec* spec) override {
        const int queryNum = 2;
        OperatorType queryOps[queryNum] = {OT_Conv, OT_FC};

        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_BatchNorm) {
                int bnOpIndex = i;
                int convOpIndex = searchOperatorIndexBackward(spec, i-1, queryOps, queryNum);
                if (convOpIndex == -1) {
                    std::cout << "[WARNING] encounter unoptimize BatchNorm layer(no Conv before): " << spec->ops[i].name << std::endl;
                    continue;
                }

                // tensor relationship rewrite
                str_copy(spec->ops[convOpIndex].output_tensors_name[0], spec->ops[bnOpIndex].output_tensors_name[0], NAME_LEN);
                hasOptimized = true;

                // bn
                int bnWeightIndex = searchWeightIndex(spec, spec->ops[bnOpIndex].name);
                assert(bnWeightIndex >= 0);
                assert(spec->ws[bnWeightIndex].mdt == DT_F32);
                F32 epsCur = spec->ops[bnOpIndex].ps.bn_param_spec.eps;
                U32 channelCur = spec->ws[bnWeightIndex].bytes_of_weight / bytesOf(spec->ws[bnWeightIndex].mdt);
                F32* meanPtr = (F32 *)spec->ws[bnWeightIndex].weight;
                F32* varPtr = (F32 *)spec->ws[bnWeightIndex].vec;

                std::vector<float> stdValue(channelCur);
                for (U32 j=0; j < channelCur; j++) {
                    stdValue[j] = sqrt(varPtr[j] + epsCur);
                }

                // conv
                int convWeightIndex = searchWeightIndex(spec, spec->ops[convOpIndex].name);
                assert(convWeightIndex >= 0);
                // Now weight mdt can be DT_DOREFA or DT_XNOR
                U32 isBNN = 0;
                if (spec->ws[convWeightIndex].mdt == DT_DOREFA || spec->ws[convWeightIndex].mdt == DT_XNOR) {
                    isBNN = 1;
                }
                F32* weightTemp = (F32*)spec->ws[convWeightIndex].weight;
                if(spec->ws[convWeightIndex].vec == nullptr){
                    spec->ws[convWeightIndex].bytes_of_vec = channelCur * sizeof(F32);
                    if (isBNN == 1) {
                        spec->ws[convWeightIndex].bytes_of_vec *= 2;
                    }
                    spec->ws[convWeightIndex].vec = (U8 *)mt_malloc(spec->ws[convWeightIndex].bytes_of_vec);
                    memset(spec->ws[convWeightIndex].vec, 0, spec->ws[convWeightIndex].bytes_of_vec);
                }
                F32* vecTemp    = (F32*)spec->ws[convWeightIndex].vec;
                if (isBNN == 1) { // Do not modify weights for BNN
                    F32* scale = vecTemp;
                    F32* bias = vecTemp + channelCur;
                    for (U32 m = 0; m < channelCur; m++) {
                        scale[m] = 1.0 / stdValue[m]; // This is the first possible source of a meaningful scale, so just initilize
                        bias[m] = (bias[m] - meanPtr[m]) / stdValue[m];
                    }
                } else {
                    int weightDataSize = spec->ws[convWeightIndex].bytes_of_weight / bytesOf(spec->ws[convWeightIndex].mdt);
                    int weightPerChannel = weightDataSize / channelCur;
                    // NCHW
                    for (U32 m = 0; m < channelCur; m++) {
                        F32* convWeightPerChannel =  weightTemp + weightPerChannel * m;
                        for (int n = 0; n < weightPerChannel; n++) {
                            convWeightPerChannel[n] /= stdValue[m];
                        }
                        vecTemp[m] = (vecTemp[m] - meanPtr[m]) / stdValue[m];
                    }
                }
                // free BN memory
                if(spec->ws[bnWeightIndex].weight != nullptr) {
                    spec->ws[bnWeightIndex].bytes_of_weight = 0;
                    free(spec->ws[bnWeightIndex].weight);
                    spec->ws[bnWeightIndex].weight = nullptr;
                }
                if(spec->ws[bnWeightIndex].vec != nullptr) {
                    spec->ws[bnWeightIndex].bytes_of_vec = 0;
                    free(spec->ws[bnWeightIndex].vec);
                    spec->ws[bnWeightIndex].vec = nullptr;
                }
                setOperatorInvalid(spec, bnOpIndex);
            }
        }
        return hasOptimized;
    }
};


class ConvScaleOptimizer: public OPOptimizer {
    virtual bool optimize(ModelSpec* spec) override {
        const int queryNum = 2;
        OperatorType queryOps[queryNum] = {OT_Conv, OT_FC};
 
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Scale) {
                int scaleOpIndex = i;
                if (spec->ops[scaleOpIndex].num_inputs > 1){
                    std::cout << "[WARNING] encounter unoptimize Scale layer(multi-inputs): " << spec->ops[i].name <<std::endl;
                    continue;
                }
                int convOpIndex = searchOperatorIndexBackward(spec, i-1, queryOps, queryNum);
                if (convOpIndex == -1) {
                    std::cout << "[WARNING] encounter unoptimize Scale layer(no Conv before): " << spec->ops[i].name <<std::endl;
                    continue;
                }

                // tensor relationship rewrite
                str_copy(spec->ops[convOpIndex].output_tensors_name[0], spec->ops[scaleOpIndex].output_tensors_name[0], NAME_LEN);
                hasOptimized = true;

                // scale
                int scaleWeightIndex = searchWeightIndex(spec, spec->ops[scaleOpIndex].name);
                assert(scaleWeightIndex >= 0);
                assert(spec->ws[scaleWeightIndex].mdt == DT_F32);
                U32 channelAlpha = spec->ws[scaleWeightIndex].bytes_of_weight / bytesOf(spec->ws[scaleWeightIndex].mdt);
                U32 channelBeta  = spec->ws[scaleWeightIndex].bytes_of_vec / bytesOf(spec->ws[scaleWeightIndex].mdt);
                U32 channelCur = std::max(channelAlpha, channelBeta);
                F32* alphaPtr = (F32*)spec->ws[scaleWeightIndex].weight;
                F32* betaPtr = (F32*)spec->ws[scaleWeightIndex].vec;

                // conv
                int convWeightIndex = searchWeightIndex(spec, spec->ops[convOpIndex].name);
                assert(convWeightIndex >= 0);
                // mdt can now be DT_DOREFA or DT_XNOR
                U32 isBNN = 0;
                if (spec->ws[convWeightIndex].mdt==DT_DOREFA || spec->ws[convWeightIndex].mdt==DT_XNOR) {
                    isBNN = 1;
                }
                F32* weightTemp = (F32*)spec->ws[convWeightIndex].weight;
                if (spec->ws[convWeightIndex].vec == nullptr) {
                    spec->ws[convWeightIndex].bytes_of_vec = channelCur * sizeof(F32);
                    if (isBNN == 1) {
                        spec->ws[convWeightIndex].bytes_of_vec *= 2;
                    }
                    spec->ws[convWeightIndex].vec = (U8 *)mt_malloc(spec->ws[convWeightIndex].bytes_of_vec);
                    memset(spec->ws[convWeightIndex].vec, 0, spec->ws[convWeightIndex].bytes_of_vec);
                }
                F32* vecTemp = (F32*)spec->ws[convWeightIndex].vec;
                if (isBNN == 1) {
                    F32 *scale = vecTemp;
                    F32 *bias = vecTemp + channelCur;
                    for (U32 m = 0; m < channelCur; m++) {
                        if (scale[m] == 0) {
                            scale[m] = alphaPtr[m];
                        } else {
                            scale[m] *= alphaPtr[m];
                        }
                        bias[m] *= alphaPtr[m];
                        if (betaPtr != nullptr) {
                            bias[m] += betaPtr[m];
                        }
                    }
                } else {
                    int weightDataSize = spec->ws[convWeightIndex].bytes_of_weight / bytesOf(spec->ws[convWeightIndex].mdt);
                    int weightPerChannel = weightDataSize / channelCur;
                    // NCHW
                    for (U32 m = 0; m < channelCur; m++){
                        F32* convWeightPerChannel =  weightTemp + weightPerChannel * m;
                        if (alphaPtr != nullptr) {
                            for(int n = 0; n < weightPerChannel; n++){
                                convWeightPerChannel[n] *= alphaPtr[m];
                            }
                            vecTemp[m] = alphaPtr[m] * vecTemp[m];
                        }
                        if (betaPtr != nullptr) {
                            vecTemp[m] += betaPtr[m];
                        }
                            
                    }
                }
                // free scale memory
                if (spec->ws[scaleWeightIndex].weight != nullptr) {
                    spec->ws[scaleWeightIndex].bytes_of_weight = 0;
                    free(spec->ws[scaleWeightIndex].weight);
                    spec->ws[scaleWeightIndex].weight = nullptr;
                }
                if(spec->ws[scaleWeightIndex].vec != nullptr) {
                    spec->ws[scaleWeightIndex].bytes_of_vec = 0;
                    free(spec->ws[scaleWeightIndex].vec);
                    spec->ws[scaleWeightIndex].vec = nullptr;
                }
                setOperatorInvalid(spec, scaleOpIndex);
            }
        }
        return hasOptimized;
    }
};


class ConvActivationOptimizer: public OPOptimizer {
    virtual bool optimize(ModelSpec* spec) override {
        const int queryNum = 1;
        OperatorType queryOps[queryNum] = {OT_Conv};
        bool hasOptimized = false;
        for (int i = 1; i< spec->num_operator_specs; i++) {
            //if (spec->ops[i].type == OT_Relu || spec->ops[i].type == OT_Relu6 || spec->ops[i].type == OT_HSwish) {
            if (spec->ops[i].type == OT_Relu) {
                if (spec->ops[i].num_inputs > 0 && spec->ops[i].num_outputs > 0) {
                    std::string inputName = spec->ops[i].input_tensors_name[0];
                    std::string outputName = spec->ops[i].output_tensors_name[0];
                    if (inputName != outputName) {
                        std::cout << "[WARNING] encounter unoptimize Relu layer(not inPlace): " << spec->ops[i].name << std::endl;
                        continue;
                    }
                }
                int atOpIndex   = i;
                int convOpIndex = searchOperatorIndexBackward(spec, atOpIndex - 1, queryOps, queryNum);

                if (convOpIndex == -1) {
                    std::cout << "[WARNING] encounter unoptimize Relu layer(no Conv before): " << spec->ops[atOpIndex].name << std::endl;
                    continue;
                }

                // tensor relationship rewrite
                str_copy(spec->ops[convOpIndex].output_tensors_name[0], spec->ops[atOpIndex].output_tensors_name[0], NAME_LEN);
                hasOptimized = true;

                switch (spec->ops[convOpIndex].ps.conv_param_spec.convolution_type) {
                    case Convolution_Pointwise:
                        spec->ops[convOpIndex].ps.conv_param_spec.pw_activation_type = ACTIVATION_RELU;
                        break;
                    case Convolution_Depthwise:
                        spec->ops[convOpIndex].ps.conv_param_spec.dw_activation_type = ACTIVATION_RELU;
                        break;
                    default:
                        assert(0);
                        break;
                }
                spec->ops[convOpIndex].ps.conv_param_spec.activation_spec.relu_spec = spec->ops[atOpIndex].ps.relu_spec;
                setOperatorInvalid(spec, atOpIndex);
            }
            if (spec->ops[i].type == OT_Sigmoid || spec->ops[i].type == OT_Clip){
                //tensor_computing not support fusion
                std::cout << "[ERROR] encounter unoptimize " << OperatorTypeName()[spec->ops[i].type] << " layer" <<std::endl;
                continue;
            }
        }
        return hasOptimized;
    }
};


class ConvConvDepthwiseOptimizer: public OPOptimizer {
    virtual bool optimize(ModelSpec* spec) override {
        const int queryNum = 1;
        OperatorType queryOps[queryNum] = {OT_Conv};

        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            // process depthwise convolution
            if (spec->ops[i].type == OT_Conv && spec->ops[i].ps.conv_param_spec.convolution_type == Convolution_Depthwise) {
                int dwConvOpIndex = i;
                int convOpIndex = searchOperatorIndexForward(spec, i + 1,  queryOps, queryNum);
                if (convOpIndex == -1 || \
                    spec->ops[convOpIndex].ps.conv_param_spec.convolution_type != Convolution_Pointwise) {
                    std::cout << "[WARNING] encounter unoptimize Depthwise Convolution(no Pointwise Convolution after): " << spec->ops[dwConvOpIndex].name << std::endl;
                    continue;
                }

                // reallocate weights and bias
                int dwConvWeightIndex = searchWeightIndex(spec, spec->ops[dwConvOpIndex].name);
                int convWeightIndex = searchWeightIndex(spec, spec->ops[convOpIndex].name);
                assert(dwConvWeightIndex != -1);
                assert(convWeightIndex != -1);
                assert(spec->ws[dwConvWeightIndex].mdt == DT_F32);
                assert(spec->ws[convWeightIndex].mdt == DT_F32);

                U32 weightSize = spec->ws[dwConvWeightIndex].bytes_of_weight + spec->ws[convWeightIndex].bytes_of_weight;
                U8* weight = (U8 *)mt_malloc(weightSize);
                memcpy(weight, spec->ws[dwConvWeightIndex].weight, spec->ws[dwConvWeightIndex].bytes_of_weight);
                memcpy(weight + spec->ws[dwConvWeightIndex].bytes_of_weight,
                       spec->ws[convWeightIndex].weight,
                       spec->ws[convWeightIndex].bytes_of_weight);

                U32 vecSize = sizeof(F32) * (spec->ops[dwConvOpIndex].ps.conv_param_spec.num_kernels \
                                           + spec->ops[convOpIndex].ps.conv_param_spec.num_kernels);
                U8* vec = (U8 *)mt_malloc(vecSize);
                U8* ptr = vec;
                if (spec->ws[dwConvWeightIndex].bytes_of_vec == 0) {
                    memset(ptr, 0, sizeof(F32)*(spec->ops[dwConvOpIndex].ps.conv_param_spec.num_kernels));
                }
                else{
                    assert(sizeof(F32)*(spec->ops[dwConvOpIndex].ps.conv_param_spec.num_kernels) == spec->ws[dwConvWeightIndex].bytes_of_vec);
                    memcpy(ptr, spec->ws[dwConvWeightIndex].vec, spec->ws[dwConvWeightIndex].bytes_of_vec);
                }
                ptr = vec + sizeof(F32)*(spec->ops[dwConvOpIndex].ps.conv_param_spec.num_kernels);
                if (spec->ws[convWeightIndex].bytes_of_vec == 0) {
                    memset(ptr, 0, sizeof(F32)*(spec->ops[convOpIndex].ps.conv_param_spec.num_kernels));
                }
                else{
                    assert(sizeof(F32)*(spec->ops[convOpIndex].ps.conv_param_spec.num_kernels) == spec->ws[convWeightIndex].bytes_of_vec);
                    memcpy(ptr, spec->ws[convWeightIndex].vec, spec->ws[convWeightIndex].bytes_of_vec);
                }

                // free and reallocate
                if(spec->ws[dwConvWeightIndex].weight != nullptr) {
                    spec->ws[dwConvWeightIndex].bytes_of_weight = 0;
                    free(spec->ws[dwConvWeightIndex].weight);
                    spec->ws[dwConvWeightIndex].weight = nullptr;
                }
                if(spec->ws[dwConvWeightIndex].vec != nullptr) {
                    spec->ws[dwConvWeightIndex].bytes_of_vec = 0;
                    free(spec->ws[dwConvWeightIndex].vec);
                    spec->ws[dwConvWeightIndex].vec = nullptr;
                }
                if(spec->ws[convWeightIndex].weight != nullptr) {
                    spec->ws[convWeightIndex].bytes_of_weight = 0;
                    free(spec->ws[convWeightIndex].weight);
                    spec->ws[convWeightIndex].weight = nullptr;
                }
                if(spec->ws[convWeightIndex].vec != nullptr) {
                    spec->ws[convWeightIndex].bytes_of_vec = 0;
                    free(spec->ws[convWeightIndex].vec);
                    spec->ws[convWeightIndex].vec = nullptr;
                }

                // retain depthwise convolution operator
                str_copy(spec->ops[dwConvOpIndex].output_tensors_name[0], spec->ops[convOpIndex].output_tensors_name[0], NAME_LEN);
                spec->ops[dwConvOpIndex].ps.conv_param_spec.num_kernels = spec->ops[convOpIndex].ps.conv_param_spec.num_kernels;
                spec->ops[dwConvOpIndex].ps.conv_param_spec.convolution_type = Convolution_Depthwise_Pointwise;
                spec->ops[dwConvOpIndex].ps.conv_param_spec.pw_activation_type = spec->ops[convOpIndex].ps.conv_param_spec.pw_activation_type;
                spec->ws[dwConvWeightIndex].bytes_of_weight = weightSize;
                spec->ws[dwConvWeightIndex].weight = weight;
                spec->ws[dwConvWeightIndex].bytes_of_vec = vecSize;
                spec->ws[dwConvWeightIndex].vec = vec;
                hasOptimized = true;

                setOperatorInvalid(spec, convOpIndex);
                i = convOpIndex;
            }
        }
        return hasOptimized;
    }
};


class ConvEltwisePoolingOptimizer: public OPOptimizer {
    virtual bool optimize(ModelSpec* spec) override {
        if(spec == nullptr)
            return false;

        bool hasOptimized = false;
        // TODO: add fusion(low priority)
        return hasOptimized;
    }
};


class FCEltwiseOptimizer: public OPOptimizer {
    virtual bool optimize(ModelSpec* spec) override {
        if(spec == nullptr)
            return false;

        bool hasOptimized = false;
        // TODO: add fusion(middle priority)
        return hasOptimized;
    }
};


class TransposeMulToScaleOptimizer: public OPOptimizer {
    virtual bool optimize(ModelSpec* spec) override {
        const int queryNum = 2;
        OperatorType queryOps[queryNum] = {OT_Transpose, OT_Reshape};
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Eltwise && spec->ops[i].ps.eltwise_param_spec.elt_mode == ELTWISE_PROD) {
                int mulOpIndex  = i;
                int transposeOpIndex00 = searchOperatorIndexBackward(spec, mulOpIndex - 1, queryOps, queryNum, false);
                if (transposeOpIndex00 == -1)
                    continue;
                int transposeOpIndex01 = searchOperatorIndexBackward(spec, transposeOpIndex00 - 1, queryOps, queryNum, false);
                if (transposeOpIndex01 == -1)
                    continue;
                int transposeOpIndex10 = searchOperatorIndexForward(spec, mulOpIndex + 1, queryOps, queryNum, false);
                if (transposeOpIndex10 == -1)
                    continue;
                
                if (transposeOpIndex10 == mulOpIndex + 1
                    || (transposeOpIndex10 == mulOpIndex + 2 && spec->ops[mulOpIndex+1].type == OT_Relu)) {
                    str_copy(spec->ops[mulOpIndex].input_tensors_name[0], spec->ops[transposeOpIndex00].input_tensors_name[0], NAME_LEN);
                    str_copy(spec->ops[mulOpIndex].input_tensors_name[1], spec->ops[transposeOpIndex01].input_tensors_name[0], NAME_LEN);
                    str_copy(spec->ops[transposeOpIndex10-1].output_tensors_name[0], spec->ops[transposeOpIndex10].output_tensors_name[0], NAME_LEN);

                    hasOptimized = true;
                    spec->ops[mulOpIndex].type = OT_Scale;

                    setOperatorInvalid(spec, transposeOpIndex00);
                    setOperatorInvalid(spec, transposeOpIndex01);
                    setOperatorInvalid(spec, transposeOpIndex10);
                    i = transposeOpIndex10;
                }
            }
        }
        return hasOptimized;
    }
};


class MergeOperatorOptimizer: public OPOptimizer {
    virtual bool optimize(ModelSpec* spec) override {
        const int queryNum = 1;
        OperatorType queryOps[queryNum] = {OT_Clip};
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Clip) {
                int opIndex0 = i;
                int opIndex1 = searchOperatorIndexForward(spec, opIndex0+1, queryOps, queryNum);
                if (opIndex1 == -1)
                    continue;

                str_copy(spec->ops[opIndex0].output_tensors_name[0], spec->ops[opIndex1].output_tensors_name[0], NAME_LEN);
                hasOptimized = true;
                spec->ops[opIndex0].ps.clip_spec.min = std::max(spec->ops[opIndex0].ps.clip_spec.min, 
                                                                  spec->ops[opIndex1].ps.clip_spec.min);
                spec->ops[opIndex0].ps.clip_spec.max = std::min(spec->ops[opIndex0].ps.clip_spec.max, 
                                                                  spec->ops[opIndex1].ps.clip_spec.max);
                setOperatorInvalid(spec, opIndex1);
                i = opIndex1;
            }
        }
        return hasOptimized;
    }
};



class DeprecatedOpOptimizer: public OPOptimizer {
    virtual bool optimize(ModelSpec* spec) override {
        bool hasOptimized = false;

        for (int i = 0; i< spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_None && i > 0) {
                str_copy(spec->ops[i - 1].output_tensors_name[0], spec->ops[i].output_tensors_name[0], NAME_LEN);
                hasOptimized = true;
                continue;
            }

            if (spec->ops[i].type == OT_Pad) {
                if(spec->ops[i].ps.pad_spec.top == 0 && spec->ops[i].ps.pad_spec.bottom == 0 && 
                    spec->ops[i].ps.pad_spec.left == 0 && spec->ops[i].ps.pad_spec.right == 0){
                    str_copy(spec->ops[i + 1].input_tensors_name[0], spec->ops[i].input_tensors_name[0], NAME_LEN);
                    hasOptimized = true;
                    spec->ops[i].type = OT_None;  // trick
                    continue;
                }
            }

            if(spec->ops[i].type == OT_Input) {
                spec->ops[i].type = OT_None;  // trick
                continue;
            }
        }
        return hasOptimized;
    }
    public:
        static bool isDeprecatedOp(OperatorType opType){
            if (opType == OT_None) {
                return true;
            } else {
                return false;
            }
        }
        static bool isDeprecatedOpWeight(const ModelSpec* spec, int index){
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
};


class ModelSpecOptimizer {
    public:
        ModelSpecOptimizer() { }
        /**
         * @param model
         */
        bool optimize(ModelSpec* spec) {
            bool optimizeOrNot = false;
            for (auto opo: opos) {
                if (opo->optimize(spec)) {
                    optimizeOrNot = true;
                }
            }
            return optimizeOrNot;
        }

        void suggest() {
            // strict order
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new DeprecatedOpOptimizer()));
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new ConvBNOptimizer()));
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new ConvScaleOptimizer()));
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new ConvActivationOptimizer()));
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new ConvConvDepthwiseOptimizer()));
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new TransposeMulToScaleOptimizer()));
            this->opos.push_back(std::shared_ptr<OPOptimizer>(new MergeOperatorOptimizer()));
        }

        void empty() {}

    private:
        // ModelSpecOptimizer() { }
        /**
         * @param opo
         */
        std::vector<std::shared_ptr<OPOptimizer>> opos;
};

#endif
