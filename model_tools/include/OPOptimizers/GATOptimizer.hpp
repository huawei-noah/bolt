// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_GATOPTIMIZER
#define _H_GATOPTIMIZER

#include <set>
#include "OPOptimizer.hpp"

class GATOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        std::set<OperatorType> types = {OT_Expand, OT_Unsqueeze, OT_Reshape, OT_Reduction,
            OT_Transpose, OT_Tile, OT_Gather, OT_None, OT_Power, OT_Eltwise, OT_Squeeze};
        int gat_layer = 0;
        for (int i = 0; i < spec->num_operator_specs - 10; i++) {
            if (spec->ops[i].type == OT_Gather && spec->ops[i + 1].type == OT_Gather &&
                spec->ops[i + 2].type == OT_FC && spec->ops[i + 4].type == OT_FC &&
                spec->ops[i + 5].type == OT_Eltwise && spec->ops[i + 6].type == OT_Eltwise &&
                spec->ops[i + 8].type == OT_Exp && spec->ops[i + 9].type == OT_Transpose) {
                ActivationParamSpec activation;
                if (spec->ops[i + 7].type == OT_Relu) {
                    activation.mode = ACTIVATION_RELU;
                    activation.value[0] = spec->ops[i + 7].ps.relu_spec.neg_slope;
                } else {
                    continue;
                }

                int k = i + 10;
                bool gat = true;
                int num_heads = -1;
                while (k < spec->num_operator_specs - 1 &&
                    !(spec->ops[k].type == OT_Transpose && spec->ops[k + 1].type == OT_MatMul)) {
                    if (types.find(spec->ops[k].type) != types.end()) {
                        k++;
                    } else if (spec->ops[k].type == OT_Scatter) {
                        num_heads++;
                        k++;
                    } else if (spec->ops[k].type == OT_Concat) {
                        k++;
                    } else {
                        gat = false;
                        break;
                    }
                }
                if (!gat || num_heads <= 0) {
                    continue;
                }

                OperatorSpec p;
                UNI_MEMSET(&p, 0, sizeof(OperatorSpec));
                std::string opName = "gat" + std::to_string(gat_layer++);
                UNI_MEMCPY(p.name, opName.c_str(), opName.size());
                p.type = OT_GAT;
                p.num_inputs = 5;
                p.input_tensors_name = (I8 **)mt_malloc(p.num_inputs * sizeof(I8 *));
                for (int j = i, n = 0; j < i + 2; j++) {
                    for (U32 k = 0; k < spec->ops[j].num_inputs; k++, n++) {
                        p.input_tensors_name[n] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
                        UNI_STRCPY(p.input_tensors_name[n], spec->ops[j].input_tensors_name[k]);
                    }
                }
                p.input_tensors_name[4] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
                UNI_STRCPY(p.input_tensors_name[4], spec->ops[i + 4].output_tensors_name[0]);
                p.num_outputs = 1;
                p.output_tensors_name = (I8 **)mt_malloc(p.num_outputs * sizeof(I8 *));
                p.output_tensors_name[0] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
                UNI_STRCPY(p.output_tensors_name[0], spec->ops[k - 1].output_tensors_name[0]);

                p.ps.gat_spec.num_heads = num_heads;
                p.ps.gat_spec.activation_type = activation;
                int n = i + 5;
                for (U32 j = 0; j < spec->ops[n].num_inputs; j++) {
                    mt_free(spec->ops[n].input_tensors_name[j]);
                }
                mt_free(spec->ops[n].input_tensors_name);
                for (U32 j = 0; j < spec->ops[n].num_outputs; j++) {
                    mt_free(spec->ops[n].output_tensors_name[j]);
                }
                mt_free(spec->ops[n].output_tensors_name);
                spec->ops[n] = p;

                setOperatorInvalid(spec, i, false);
                setOperatorInvalid(spec, i + 1, false);
                for (int j = n + 1; j < k; j++) {
                    setOperatorInvalid(spec, j, false);
                }
                hasOptimized = true;
                i = k;
            }
        }
        return hasOptimized;
    }
};
#endif
