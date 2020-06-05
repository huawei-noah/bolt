// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_NOQUANTLABELOPTIMIZER
#define _H_NOQUANTLABELOPTIMIZER

#include <vector>
#include <string>
#include "model_tools.h"
#include "OPOptimizer.hpp"

class NoQuantLabelOptimizer: public OPOptimizer {
public:
    NoQuantLabelOptimizer(float clipVal)
    {
        if (clipVal > 0) {
            this->uniScale = 127.0 / clipVal;
        } else {
            this->uniScale = 0;
        }
    }

    virtual bool optimize(ModelSpec* spec) override
    {
        bool hasOptimized = false;
        bool firstOpChecked = false;

        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_None) {
                continue;
            }

            if (uniScale > 0) {
                if (spec->ops[i].type == OT_FC || spec->ops[i].type == OT_MatMul || spec->ops[i].type == OT_LSTM) {
                    this->label_clip_input(spec->ops + i);
                    if (spec->ops[i].type == OT_FC || spec->ops[i].type == OT_LSTM) {
                        int weightIdx = searchWeightIndex(spec, spec->ops[i].name);
                        CHECK_REQUIREMENT(-1 != weightIdx);
                        CHECK_REQUIREMENT(DT_F32 == spec->ws[weightIdx].mdt);
                        std::cout << "[INFO] Clipping the weight of FC or LSTM\n";
                        F32 clipMax = 127.0 / uniScale;
                        F32 clipMin = -1 * clipMax;
                        U32 len = spec->ws[weightIdx].bytes_of_weight / bytesOf(DT_F32);
                        F32 *w = (F32*)spec->ws[weightIdx].weight;
                        for (U32 j = 0; j < len; j++) {
                            if (w[j] > clipMax) {
                                w[j] = clipMax;
                            } else if (w[j] < clipMin) {
                                w[j] = clipMin;
                            }
                        }
                    }
                }
                continue;
            }

            if (!firstOpChecked) {
                if (OT_Conv == spec->ops[i].type) {
                    this->label_OP_as_no_quant(spec->ops + i);
                    hasOptimized = true;
                }
                firstOpChecked = true;
            }

            if (spec->ops[i].type == OT_Eltwise) {
                for (U32 j = 0; j < spec->ops[i].num_inputs; j++) {
                    std::string curIn = spec->ops[i].input_tensors_name[j];
                    this->label_fp_outputs(spec, curIn);
                    hasOptimized = true;
                }
            }

            // Activation other than ReLU
            if (spec->ops[i].type == OT_Relu6 ||
                spec->ops[i].type == OT_HSwish ||
                spec->ops[i].type == OT_HSigmoid ||
                spec->ops[i].type == OT_Sigmoid ||
                spec->ops[i].type == OT_Clip ||
                spec->ops[i].type == OT_Gelu ||
                spec->ops[i].type == OT_TanH) {
                
                std::string curIn = spec->ops[i].input_tensors_name[0];
                this->label_fp_outputs(spec, curIn);
                hasOptimized = true;
            }

            if (spec->ops[i].type == OT_Concat) {
                for (U32 j = 0; j < spec->ops[i].num_inputs; j++) {
                    std::string curIn = spec->ops[i].input_tensors_name[j];
                    int prevIndex = searchOperatorIndexByOutput(spec, curIn);
                    if (-1 == prevIndex) {  // model input
                        std::string outName = spec->ops[i].output_tensors_name[0];
                        this->label_fp_outputs(spec, outName);
                        break;
                    }
                }
            }

            if (spec->ops[i].type == OT_Softmax) {
                std::string output = spec->ops[i].output_tensors_name[0];
                bool isModelOutput = false;

                for (int j = 0; j < spec->num_outputs; j++) {
                    std::string name = spec->output_names[j];
                    if (name == output) {
                        isModelOutput = true;
                        break;
                    }
                }

                if (isModelOutput) {
                    std::string inputName = spec->ops[i].input_tensors_name[0];
                    int prevKeyIndex = searchOperatorIndexByOutput(spec, inputName);
                    while (-1 != prevKeyIndex) {
                        OperatorType ot = spec->ops[prevKeyIndex].type;
                        if (OT_Conv == ot || OT_FC == ot || OT_MatMul == ot) {
                            break;
                        } else {
                            inputName = spec->ops[prevKeyIndex].input_tensors_name[0];
                            prevKeyIndex = searchOperatorIndexByOutput(spec, inputName);
                        }
                    }

                    if (-1 == prevKeyIndex) {
                        std::cout << "[INFO] Softmax receives model input directly\n";
                        continue;
                    }
                    this->label_OP_as_no_quant(spec->ops + prevKeyIndex);

                    std::string prevIn = spec->ops[prevKeyIndex].input_tensors_name[0];
                    this->label_fp_outputs(spec, prevIn);
                } else {
                    std::string curIn = spec->ops[i].input_tensors_name[0];
                    this->label_fp_outputs(spec, curIn);
                }
                hasOptimized = true;
            }
        }
        return hasOptimized;
    }

    float uniScale;

    static void label_OP_as_no_quant(OperatorSpec* ptr)
    {
        CHECK_REQUIREMENT(0 == ptr->num_quant_feature);
        ptr->num_quant_feature = 1;
        ptr->feature_scale = (QuantSpec*)mt_new_storage(sizeof(QuantSpec));
        ptr->feature_scale[0].num_scale = 1;
        ptr->feature_scale[0].scale = (F32*)mt_new_storage(sizeof(F32));
        ptr->feature_scale[0].scale[0] = 0;
    }

    void label_fp_outputs(ModelSpec* ms, std::string tensorName)
    {
        int prevIndex = searchOperatorIndexByOutput(ms, tensorName);
        if (-1 == prevIndex) {
            return;
        }
        OperatorSpec *ptr = ms->ops + prevIndex;
        if (0 == ptr->num_quant_feature) {
            ptr->num_quant_feature = 1;
            ptr->feature_scale = (QuantSpec*)mt_new_storage(sizeof(QuantSpec));
            ptr->feature_scale[0].num_scale = 1;
            ptr->feature_scale[0].scale = (F32*)mt_new_storage(sizeof(F32));
            ptr->feature_scale[0].scale[0] = -2;
        } else if (-2 == ptr->feature_scale[0].scale[0] || 0 == ptr->feature_scale[0].scale[0]) {
            return;  // Already processed the upstream
        }

        OperatorType ot = ms->ops[prevIndex].type;
        if (OT_Conv != ot && OT_FC != ot && OT_MatMul != ot) {
            for (U32 i = 0; i < ms->ops[prevIndex].num_inputs; i++) {
                std::string name = ms->ops[prevIndex].input_tensors_name[i];
                label_fp_outputs(ms, name);
            }
        }
    }

    void label_clip_input(OperatorSpec* ptr)
    {
        CHECK_REQUIREMENT(0 == ptr->num_quant_feature);
        ptr->num_quant_feature = ptr->num_inputs + ptr->num_outputs;
        ptr->feature_scale = (QuantSpec*)mt_new_storage(sizeof(QuantSpec) * ptr->num_quant_feature);
        U32 i;
        for (i = 0; i < ptr->num_inputs; i++) {
            ptr->feature_scale[i].num_scale = 1;
            ptr->feature_scale[i].scale = (F32*)mt_new_storage(sizeof(F32));
            ptr->feature_scale[i].scale[0] = this->uniScale;
        }
        for (; i < ptr->num_quant_feature; i++) {
            ptr->feature_scale[i].num_scale = 1;
            ptr->feature_scale[i].scale = (F32*)mt_new_storage(sizeof(F32));
            ptr->feature_scale[i].scale[0] = -2;
        }
    }
};
#endif
