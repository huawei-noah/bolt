// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_CONVCONVOPTIMIZER
#define _H_CONVCONVOPTIMIZER

#include "OPOptimizer.hpp"

class ConvConvOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        hasOptimized |= horizontal_optimize(spec);
        return hasOptimized;
    }

    bool horizontal_optimize(ModelSpec *spec)
    {
        bool hasOptimized = false;
        for (int i = 2; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_Concat && spec->ops[i].ps.concat_spec.axis == 1) {
                std::vector<int> convOps;
                std::vector<int> convWeights;
                std::set<std::string> convInputs;
                std::set<std::string> convParams;
                int weightSize = 0;
                int vecSize = 0;
                int channels = 0;
                bool fuseConv = true;
                for (unsigned int j = 0; j < spec->ops[i].num_inputs; j++) {
                    auto prevOps =
                        searchOperatorIndexByOutput(spec, spec->ops[i].input_tensors_name[j], 0, i);
                    if (prevOps.size() != 1) {
                        fuseConv = false;
                        break;
                    }
                    int k = prevOps[0].first;
                    if (spec->ops[k].type == OT_Conv &&
                        (spec->ops[k].ps.conv_spec.convolution_type == Convolution_Pointwise ||
                            spec->ops[k].ps.conv_spec.convolution_type == Convolution_Dilation)) {
                        int id = searchWeightIndex(spec, spec->ops[k].name);
                        auto nextOps = searchOperatorIndexByInput(spec,
                            spec->ops[i].input_tensors_name[j], k + 1, spec->num_operator_specs);
                        if (id < 0 || spec->ws[id].mdt != DT_F32 ||
                            spec->ws[id].bytes_of_weight == 0 || spec->ws[id].bytes_of_vec == 0 ||
                            nextOps.size() != 1) {
                            fuseConv = false;
                            break;
                        }
                        convInputs.insert(spec->ops[k].input_tensors_name[0]);
                        convOps.push_back(k);
                        convWeights.push_back(id);
                        convParams.insert(
                            copyBuffer<1024>(&(spec->ops[k].ps), sizeof(ParameterSpec)));
                        weightSize += spec->ws[id].bytes_of_weight;
                        vecSize += spec->ws[id].bytes_of_vec;
                        channels += spec->ops[k].ps.conv_spec.num_outputs;
                    } else {
                        fuseConv = false;
                        break;
                    }
                }
                if (fuseConv && convInputs.size() == 1 && convParams.size() == 1 &&
                    convOps.size() > 1) {
                    U8 *weight = (U8 *)mt_new_storage(weightSize);
                    U8 *vec = (U8 *)mt_new_storage(vecSize);
                    int id = convWeights[0];
                    memcpy(weight, spec->ws[id].weight, spec->ws[id].bytes_of_weight);
                    if (outOfFileMapRange(spec->ws[id].weight, spec->mfd)) {
                        delete spec->ws[id].weight;
                    }
                    spec->ws[id].weight = weight;
                    weight += spec->ws[id].bytes_of_weight;
                    spec->ws[id].bytes_of_weight = weightSize;
                    memcpy(vec, spec->ws[id].vec, spec->ws[id].bytes_of_vec);
                    if (outOfFileMapRange(spec->ws[id].vec, spec->mfd)) {
                        delete spec->ws[id].vec;
                    }
                    spec->ws[id].vec = vec;
                    vec += spec->ws[id].bytes_of_vec;
                    spec->ws[id].bytes_of_vec = vecSize;
                    spec->ops[convOps[0]].ps.conv_spec.num_outputs = channels;
                    for (unsigned int j = 1; j < convOps.size(); j++) {
                        int id = convWeights[j];
                        memcpy(weight, spec->ws[id].weight, spec->ws[id].bytes_of_weight);
                        weight += spec->ws[id].bytes_of_weight;
                        memcpy(vec, spec->ws[id].vec, spec->ws[id].bytes_of_vec);
                        vec += spec->ws[id].bytes_of_vec;

                        setOperatorInvalid(spec, convOps[j]);
                    }
                    setOperatorInvalid(spec, i, true);
                    hasOptimized = true;
                }
            }
        }
        return hasOptimized;
    }
};
#endif
