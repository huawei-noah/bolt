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
        hasOptimized |= conv_concat_optimize(spec);
        //hasOptimized |= conv_conv_optimize(spec);
        return hasOptimized;
    }

    bool conv_conv_optimize(ModelSpec *spec)
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs - 1; i++) {
            if (spec->ops[i].num_outputs > 0) {
                auto nextOps = searchOperatorIndexByInput(
                    spec, spec->ops[i].output_tensors_name[0], i + 1, spec->num_operator_specs);
                if (nextOps.size() <= 1) {
                    continue;
                }
                int weightSize = 0;
                int vecSize = 0;
                std::vector<int> convOps;
                std::vector<int> convWeights;
                std::set<std::string> convParams;
                for (U32 j = 0; j < nextOps.size(); j++) {
                    int k = nextOps[j].first;
                    if (spec->ops[k].type == OT_Conv &&
                        (spec->ops[k].ps.conv_spec.convolution_type == CONVOLUTION_POINTWISE ||
                            spec->ops[k].ps.conv_spec.convolution_type == CONVOLUTION_DILATION)) {
                        int id = searchWeightIndex(spec, spec->ops[k].name);
                        if (id < 0 || spec->ws[id].mdt != DT_F32 ||
                            spec->ws[id].bytes_of_weight == 0 || spec->ws[id].bytes_of_vec == 0) {
                            continue;
                        }
                        convOps.push_back(k);
                        convWeights.push_back(id);
                        convParams.insert(copyBuffer<1024>(&(spec->ops[k].ps),
                            get_operator_parameter_size(sg_boltVersion, spec->ops[k].type)));
                        weightSize += spec->ws[id].bytes_of_weight;
                        vecSize += spec->ws[id].bytes_of_vec;
                    }
                }

                if (convOps.size() <= 1 || convParams.size() != 1) {
                    continue;
                }
                U8 *weight = (U8 *)mt_malloc(weightSize);
                U8 *vec = (U8 *)mt_malloc(vecSize);
                int k = convOps[0];
                int id = convWeights[0];
                UNI_MEMCPY(weight, spec->ws[id].weight, spec->ws[id].bytes_of_weight);
                mt_free(spec->ws[id].weight, spec);
                spec->ws[id].weight = weight;
                weight += spec->ws[id].bytes_of_weight;
                spec->ws[id].bytes_of_weight = weightSize;
                UNI_MEMCPY(vec, spec->ws[id].vec, spec->ws[id].bytes_of_vec);
                mt_free(spec->ws[id].vec, spec);
                spec->ws[id].vec = vec;
                vec += spec->ws[id].bytes_of_vec;
                spec->ws[id].bytes_of_vec = vecSize;

                std::string name = allocName("slice_" + std::to_string(i));
                OperatorSpec sliceOperator =
                    mt_create_operator(name.c_str(), OT_Slice, 1, convOps.size());
                UNI_STRCPY(sliceOperator.input_tensors_name[0], name.c_str());
                UNI_STRCPY(sliceOperator.output_tensors_name[0], spec->ops[k].output_tensors_name[0]);
                UNI_STRCPY(spec->ops[k].output_tensors_name[0], name.c_str());
                sliceOperator.ps.slice_spec.axis = 1;
                sliceOperator.ps.slice_spec.num_slice = convOps.size() - 1;
                sliceOperator.ps.slice_spec.slice_points[0] = spec->ops[k].ps.conv_spec.num_outputs;
                for (unsigned int j = 1; j < convOps.size(); j++) {
                    int k = convOps[j];
                    sliceOperator.ps.slice_spec.slice_points[j] =
                        sliceOperator.ps.slice_spec.slice_points[j - 1] +
                        spec->ops[k].ps.conv_spec.num_outputs;
                    UNI_STRCPY(
                        sliceOperator.output_tensors_name[j], spec->ops[k].output_tensors_name[0]);
                    spec->ops[convOps[0]].ps.conv_spec.num_outputs +=
                        spec->ops[k].ps.conv_spec.num_outputs;

                    int id = convWeights[j];
                    UNI_MEMCPY(weight, spec->ws[id].weight, spec->ws[id].bytes_of_weight);
                    weight += spec->ws[id].bytes_of_weight;
                    UNI_MEMCPY(vec, spec->ws[id].vec, spec->ws[id].bytes_of_vec);
                    vec += spec->ws[id].bytes_of_vec;

                    setOperatorInvalid(spec, k, false);
                    hasOptimized = true;
                }
                mt_insert_operator(spec, convOps[0] + 1, sliceOperator);
            }
        }
        return hasOptimized;
    }

    bool conv_concat_optimize(ModelSpec *spec)
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
                        (spec->ops[k].ps.conv_spec.convolution_type == CONVOLUTION_POINTWISE ||
                            spec->ops[k].ps.conv_spec.convolution_type == CONVOLUTION_DILATION)) {
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
                        convParams.insert(copyBuffer<1024>(&(spec->ops[k].ps),
                            get_operator_parameter_size(sg_boltVersion, spec->ops[k].type)));
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
                    U8 *weight = (U8 *)mt_malloc(weightSize);
                    U8 *vec = (U8 *)mt_malloc(vecSize);
                    int id = convWeights[0];
                    UNI_MEMCPY(weight, spec->ws[id].weight, spec->ws[id].bytes_of_weight);
                    mt_free(spec->ws[id].weight, spec);
                    spec->ws[id].weight = weight;
                    weight += spec->ws[id].bytes_of_weight;
                    spec->ws[id].bytes_of_weight = weightSize;
                    UNI_MEMCPY(vec, spec->ws[id].vec, spec->ws[id].bytes_of_vec);
                    mt_free(spec->ws[id].vec, spec);
                    spec->ws[id].vec = vec;
                    vec += spec->ws[id].bytes_of_vec;
                    spec->ws[id].bytes_of_vec = vecSize;
                    spec->ops[convOps[0]].ps.conv_spec.num_outputs = channels;
                    for (unsigned int j = 1; j < convOps.size(); j++) {
                        int id = convWeights[j];
                        UNI_MEMCPY(weight, spec->ws[id].weight, spec->ws[id].bytes_of_weight);
                        weight += spec->ws[id].bytes_of_weight;
                        UNI_MEMCPY(vec, spec->ws[id].vec, spec->ws[id].bytes_of_vec);
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
