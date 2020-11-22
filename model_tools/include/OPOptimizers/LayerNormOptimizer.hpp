// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_LayerNormOPTIMIZER
#define _H_LayerNormOPTIMIZER

#include <string>
#include <map>
#include "model_tools.h"
#include "OPOptimizer.hpp"

class LayerNormOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Power && spec->ops[i].ps.power_spec.power == 2) {
                int powOpIndex = i;
                int backAddIndex = -1;
                for (int j = powOpIndex - 1; j >= 0; j--) {
                    if (spec->ops[j].type == OT_Eltwise) {
                        if (spec->ops[j].ps.eltwise_spec.elt_mode == ELTWISE_SUM) {
                            backAddIndex = j;
                            break;
                        } else {
                            continue;
                        }
                    }
                }

                if (backAddIndex == -1) {
                    continue;
                }

                int forDivIndex = -1;
                for (int j = powOpIndex + 1; j < spec->num_operator_specs; j++) {
                    if (spec->ops[j].type == OT_Eltwise) {
                        if (spec->ops[j].ps.eltwise_spec.elt_mode == ELTWISE_DIV) {
                            forDivIndex = j;
                            break;
                        } else {
                            continue;
                        }
                    }
                }

                if (forDivIndex == -1) {
                    continue;
                }

                std::map<std::string, int> info_map;
                bool tag = true;
                for (int k = backAddIndex + 1; k < forDivIndex; k++) {
                    std::string tmp_str = "";
                    if (spec->ops[k].type == OT_Pooling) {
                        tmp_str = "ReduceMean";
                    } else if (spec->ops[k].type == OT_Eltwise) {
                        tmp_str = "Sub";
                    } else if (spec->ops[k].type == OT_Scale) {
                        tmp_str = "Scale";
                    } else if (spec->ops[k].type == OT_Power &&
                        spec->ops[k].ps.power_spec.power == 2) {
                        tmp_str = "Pow";
                    } else if (spec->ops[k].type == OT_Power &&
                        spec->ops[k].ps.power_spec.power == 0.5) {
                        tmp_str = "Sqrt";
                    } else {
                        tag = false;
                        break;
                    }

                    if (info_map.find(tmp_str) == info_map.end()) {
                        info_map[tmp_str] = 1;
                    } else {
                        info_map[tmp_str] = info_map[tmp_str] + 1;
                    }
                }

                if (tag == false) {
                    continue;
                }

                if (info_map["ReduceMean"] == 2 && info_map["Sub"] == 2 && info_map["Scale"] == 1 &&
                    info_map["Pow"] == 1 && info_map["Sqrt"] == 1 &&
                    spec->ops[forDivIndex + 1].type == OT_Scale &&
                    spec->ops[forDivIndex + 2].type == OT_Scale) {
                    hasOptimized = true;
                } else {
                    continue;
                }

                int tailMulIndex = forDivIndex + 1;
                int tailAddIndex = forDivIndex + 2;
                spec->ops[tailMulIndex].type = OT_LayerNorm;
                int tailMulWeightIndex = searchWeightIndex(spec, spec->ops[tailMulIndex].name);
                int tailAddWeightIndex = searchWeightIndex(spec, spec->ops[tailAddIndex].name);
                CHECK_REQUIREMENT(tailAddWeightIndex >= 0);
                CHECK_REQUIREMENT(spec->ws[tailAddWeightIndex].mdt == DT_F32);

                spec->ws[tailMulWeightIndex].bytes_of_vec =
                    spec->ws[tailAddWeightIndex].bytes_of_vec;
                U8 *ln_vec = (U8 *)mt_new_storage(spec->ws[tailAddWeightIndex].bytes_of_vec);
                memcpy(ln_vec, spec->ws[tailAddWeightIndex].vec,
                    spec->ws[tailAddWeightIndex].bytes_of_vec);
                spec->ws[tailMulWeightIndex].vec = ln_vec;

                if (spec->ws[tailAddWeightIndex].weight != nullptr) {
                    spec->ws[tailAddWeightIndex].bytes_of_weight = 0;
                    delete spec->ws[tailAddWeightIndex].weight;
                    spec->ws[tailAddWeightIndex].weight = nullptr;
                }

                if (spec->ws[tailAddWeightIndex].vec != nullptr) {
                    spec->ws[tailAddWeightIndex].bytes_of_vec = 0;
                    delete spec->ws[tailAddWeightIndex].vec;
                    spec->ws[tailAddWeightIndex].vec = nullptr;
                }

                memcpy(spec->ops[tailMulIndex].output_tensors_name[0],
                    spec->ops[tailAddIndex].output_tensors_name[0], NAME_LEN);
                memcpy(spec->ops[backAddIndex].output_tensors_name[0],
                    spec->ops[tailMulIndex].input_tensors_name[0], NAME_LEN);

                for (int k = backAddIndex + 1; k <= forDivIndex; k++) {
                    setOperatorInvalid(spec, k);
                }
                setOperatorInvalid(spec, tailAddIndex);

                int AddScaleWeightIndex = searchWeightIndex(spec, spec->ops[forDivIndex - 2].name);
                spec->ws[AddScaleWeightIndex].bytes_of_vec = 0;
                delete spec->ws[AddScaleWeightIndex].vec;
                spec->ws[AddScaleWeightIndex].vec = nullptr;
            }
        }
        return hasOptimized;
    }
};
#endif
