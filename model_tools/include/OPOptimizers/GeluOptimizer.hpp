// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_GeluOPTIMIZER
#define _H_GeluOPTIMIZER

#include "OPOptimizer.hpp"

class GeluOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs - 2; i++) {
            if (spec->ops[i].type == OT_Erf) {
                int erfIndex = i;
                int divIndex = erfIndex - 1;
                int AddIndex = erfIndex + 1;
                int secMulIndex = erfIndex + 2;

                int firMulIndex = -1;
                int candidate = erfIndex - 2;
                if (candidate >= 0 && spec->ops[candidate].type == OT_Power &&
                    UNI_ABS(spec->ops[candidate].ps.power_spec.scale - 0.5) < eps &&
                    UNI_ABS(spec->ops[candidate].ps.power_spec.shift) < eps &&
                    UNI_ABS(spec->ops[candidate].ps.power_spec.power - 1) < eps) {
                    firMulIndex = candidate;
                }
                candidate = erfIndex + 3;
                if (candidate < spec->num_operator_specs && spec->ops[candidate].type == OT_Power &&
                    UNI_ABS(spec->ops[candidate].ps.power_spec.scale - 0.5) < eps &&
                    UNI_ABS(spec->ops[candidate].ps.power_spec.shift) < eps &&
                    UNI_ABS(spec->ops[candidate].ps.power_spec.power - 1) < eps) {
                    firMulIndex = candidate;
                }
                if (firMulIndex == -1) {
                    continue;
                }
                if (spec->ops[divIndex].type == OT_Power &&
                    UNI_ABS(spec->ops[divIndex].ps.power_spec.scale - 1 / 1.4142135381) < eps &&
                    UNI_ABS(spec->ops[divIndex].ps.power_spec.shift) < eps &&
                    UNI_ABS(spec->ops[divIndex].ps.power_spec.power - 1) < eps) {
                    if (spec->ops[AddIndex].type == OT_Power &&
                        UNI_ABS(spec->ops[AddIndex].ps.power_spec.scale - 1) < eps &&
                        UNI_ABS(spec->ops[AddIndex].ps.power_spec.shift - 1) < eps &&
                        UNI_ABS(spec->ops[AddIndex].ps.power_spec.power - 1) < eps) {
                        if (spec->ops[secMulIndex].type == OT_Eltwise &&
                            spec->ops[secMulIndex].ps.eltwise_spec.mode == ELTWISE_PROD) {
                            spec->ops[secMulIndex].num_inputs = 1;
                            mt_free(spec->ops[secMulIndex].input_tensors_name[1]);
                            UNI_MEMCPY(spec->ops[secMulIndex].input_tensors_name[0],
                                spec->ops[divIndex].input_tensors_name[0], NAME_LEN);
                            spec->ops[secMulIndex].type = OT_Gelu;
                            setOperatorInvalid(spec, firMulIndex, true);
                            setOperatorInvalid(spec, divIndex, true);
                            setOperatorInvalid(spec, erfIndex, true);
                            setOperatorInvalid(spec, AddIndex, true);

                            hasOptimized = true;
                        }
                    }
                }
            }
        }
        return hasOptimized;
    }

private:
    float eps = 0.0001;
};
#endif
