// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_FCFCOPTIMIZER
#define _H_FCFCOPTIMIZER

#include "OPOptimizer.hpp"

class FCFCOptimizer : public OPOptimizer {
    bool optimize(ModelSpec *spec) override
    {
        bool hasOptimized = false;
        hasOptimized |= horizontal_optimize(spec);
        hasOptimized |= vertical_optimize(spec);
        return hasOptimized;
    }

    template <typename T>
    void mmm_nt(T *A, T *B, T *bias, T *C, int m, int n, int k)
    {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                T value = 0;
                for (int z = 0; z < k; z++) {
                    value += A[i * k + z] * B[j * k + z];
                }
                if (bias != nullptr) {
                    value += bias[i * n + j];
                }
                C[i * n + j] = value;
            }
        }
    }

    bool vertical_optimize(ModelSpec *spec)
    {
        bool hasOptimized = false;
        for (int i = 0; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_FC && spec->ops[i].ps.fc_spec.num_slices == 1) {
                std::vector<std::pair<int, int>> nextOpIndexes = searchOperatorIndexByInput(
                    spec, spec->ops[i].output_tensors_name[0], i, spec->num_operator_specs);
                // 1 -> N || next operator not FC
                if (nextOpIndexes.size() != 1 || spec->ops[nextOpIndexes[0].first].type != OT_FC) {
                    continue;
                }
                // y = Wa2 * (Wa1 * x + b1) + b2
                // y = Wa3 * x + b3
                // Wa3 = Wa2 * Wa1, b3 = Wa2 * b1 + b2
                int a1_id = searchWeightIndex(spec, spec->ops[i].name);
                U32 a1_m = spec->ops[i].ps.fc_spec.num_outputs;
                U32 a1_k = spec->ws[a1_id].bytes_of_weight / bytesOf(spec->ws[a1_id].mdt) / a1_m;
                int a2_id = searchWeightIndex(spec, spec->ops[nextOpIndexes[0].first].name);
                U32 a2_m = spec->ops[nextOpIndexes[0].first].ps.fc_spec.num_outputs;
                U32 a2_k = spec->ws[a2_id].bytes_of_weight / bytesOf(spec->ws[a2_id].mdt) / a2_m;
                CHECK_REQUIREMENT(a2_k == a1_m);
                int a3_id = a2_id;
                U32 a3_m = a2_m;
                U32 a3_k = a1_k;
                U32 a3_size = a3_m * a3_k * bytesOf(spec->ws[a3_id].mdt);
                U32 b3_size = a3_m * bytesOf(spec->ws[a3_id].mdt);
                U8 *a3 = (U8 *)mt_new_storage(a3_size);
                U8 *b3 = (U8 *)mt_new_storage(b3_size);
                mmm_nt<F32>((F32 *)spec->ws[a2_id].weight, (F32 *)spec->ws[a1_id].weight, nullptr,
                    (F32 *)a3, a3_m, a3_k, a2_k);
                mmm_nt<F32>((F32 *)spec->ws[a2_id].weight, (F32 *)spec->ws[a1_id].vec,
                    (F32 *)spec->ws[a2_id].vec, (F32 *)b3, a3_m, 1, a2_k);

                //erase first fc parameter
                if (spec->ws[a1_id].weight != nullptr) {
                    spec->ws[a1_id].bytes_of_weight = 0;
                    if (outOfFileMapRange(spec->ws[a1_id].weight, spec->mfd)) {
                        delete spec->ws[a1_id].weight;
                    }
                    spec->ws[a1_id].weight = nullptr;
                }
                if (spec->ws[a1_id].vec != nullptr) {
                    spec->ws[a1_id].bytes_of_vec = 0;
                    if (outOfFileMapRange(spec->ws[a1_id].vec, spec->mfd)) {
                        delete spec->ws[a1_id].vec;
                    }
                    spec->ws[a1_id].vec = nullptr;
                }
                str_copy(spec->ops[nextOpIndexes[0].first].input_tensors_name[0],
                    spec->ops[i].input_tensors_name[0], NAME_LEN);
                setOperatorInvalid(spec, i);

                if (spec->ws[a2_id].weight != nullptr &&
                    outOfFileMapRange(spec->ws[a2_id].weight, spec->mfd)) {
                    delete spec->ws[a2_id].weight;
                }
                if (spec->ws[a2_id].vec != nullptr &&
                    outOfFileMapRange(spec->ws[a2_id].vec, spec->mfd)) {
                    delete spec->ws[a2_id].vec;
                }
                spec->ws[a2_id].bytes_of_weight = a3_size;
                spec->ws[a2_id].weight = a3;
                spec->ws[a2_id].bytes_of_vec = b3_size;
                spec->ws[a2_id].vec = b3;
                hasOptimized = true;
            }
        }
        return hasOptimized;
    }

    bool horizontal_optimize(ModelSpec *spec)
    {
        const int queryNum = 1;
        OperatorType queryOps[queryNum] = {OT_FC};
        bool hasOptimized = false;
        for (int i = 1; i < spec->num_operator_specs; i++) {
            if (spec->ops[i].type == OT_FC) {
                int curOpIndex = i;
                int prevOpIndex =
                    searchOperatorIndexBackward(spec, curOpIndex - 1, queryOps, queryNum);
                if (prevOpIndex == -1) {
                    continue;
                }
                if (strncmp(spec->ops[curOpIndex].input_tensors_name[0],
                        spec->ops[prevOpIndex].input_tensors_name[0], NAME_LEN)) {
                    continue;
                }

                int prevWeightIndex = searchWeightIndex(spec, spec->ops[prevOpIndex].name);
                int curWeightIndex = searchWeightIndex(spec, spec->ops[curOpIndex].name);
                CHECK_REQUIREMENT(prevWeightIndex != -1);
                CHECK_REQUIREMENT(curWeightIndex != -1);
                CHECK_REQUIREMENT(spec->ws[prevWeightIndex].mdt == DT_F32);
                CHECK_REQUIREMENT(spec->ws[curWeightIndex].mdt == DT_F32);

                U32 weightSize = spec->ws[prevWeightIndex].bytes_of_weight +
                    spec->ws[curWeightIndex].bytes_of_weight;
                U8 *weight = (U8 *)mt_new_storage(weightSize);
                memcpy(weight, spec->ws[prevWeightIndex].weight,
                    spec->ws[prevWeightIndex].bytes_of_weight);
                memcpy(weight + spec->ws[prevWeightIndex].bytes_of_weight,
                    spec->ws[curWeightIndex].weight, spec->ws[curWeightIndex].bytes_of_weight);

                U32 vecSize = sizeof(F32) *
                    (spec->ops[prevOpIndex].ps.fc_spec.num_outputs +
                        spec->ops[curOpIndex].ps.fc_spec.num_outputs);
                U8 *vec = (U8 *)mt_new_storage(vecSize);
                U8 *ptr = vec;
                if (spec->ws[prevWeightIndex].bytes_of_vec == 0) {
                    memset(ptr, 0, sizeof(F32) * (spec->ops[prevOpIndex].ps.fc_spec.num_outputs));
                } else {
                    CHECK_REQUIREMENT(sizeof(F32) * (spec->ops[prevOpIndex].ps.fc_spec.num_outputs) ==
                        spec->ws[prevWeightIndex].bytes_of_vec);
                    memcpy(
                        ptr, spec->ws[prevWeightIndex].vec, spec->ws[prevWeightIndex].bytes_of_vec);
                }
                ptr = vec + sizeof(F32) * (spec->ops[prevOpIndex].ps.fc_spec.num_outputs);
                if (spec->ws[curWeightIndex].bytes_of_vec == 0) {
                    memset(ptr, 0, sizeof(F32) * (spec->ops[curOpIndex].ps.fc_spec.num_outputs));
                } else {
                    CHECK_REQUIREMENT(sizeof(F32) * (spec->ops[curOpIndex].ps.fc_spec.num_outputs) ==
                        spec->ws[curWeightIndex].bytes_of_vec);
                    memcpy(ptr, spec->ws[curWeightIndex].vec, spec->ws[curWeightIndex].bytes_of_vec);
                }

                if (spec->ws[prevWeightIndex].weight != nullptr) {
                    spec->ws[prevWeightIndex].bytes_of_weight = 0;
                    if (outOfFileMapRange(spec->ws[prevWeightIndex].weight, spec->mfd)) {
                        delete spec->ws[prevWeightIndex].weight;
                    }
                    spec->ws[prevWeightIndex].weight = nullptr;
                }
                if (spec->ws[prevWeightIndex].vec != nullptr) {
                    spec->ws[prevWeightIndex].bytes_of_vec = 0;
                    if (outOfFileMapRange(spec->ws[prevWeightIndex].vec, spec->mfd)) {
                        delete spec->ws[prevWeightIndex].vec;
                    }
                    spec->ws[prevWeightIndex].vec = nullptr;
                }
                if (spec->ws[curWeightIndex].weight != nullptr) {
                    spec->ws[curWeightIndex].bytes_of_weight = 0;
                    if (outOfFileMapRange(spec->ws[curWeightIndex].weight, spec->mfd)) {
                        delete spec->ws[curWeightIndex].weight;
                    }
                    spec->ws[curWeightIndex].weight = nullptr;
                }
                if (spec->ws[curWeightIndex].vec != nullptr) {
                    spec->ws[curWeightIndex].bytes_of_vec = 0;
                    if (outOfFileMapRange(spec->ws[curWeightIndex].vec, spec->mfd)) {
                        delete spec->ws[curWeightIndex].vec;
                    }
                    spec->ws[curWeightIndex].vec = nullptr;
                }

                // FC params
                spec->ops[prevOpIndex].ps.fc_spec.num_slices++;
                U32 slices = spec->ops[prevOpIndex].ps.fc_spec.num_slices;
                CHECK_REQUIREMENT(
                    slices <= sizeof(spec->ops[prevOpIndex].ps.fc_spec.slice_point) / sizeof(int));
                spec->ops[prevOpIndex].ps.fc_spec.slice_point[slices - 1] =
                    spec->ops[curOpIndex].ps.fc_spec.num_outputs;
                spec->ops[prevOpIndex].ps.fc_spec.num_outputs +=
                    spec->ops[curOpIndex].ps.fc_spec.num_outputs;

                // operator spec
                spec->ops[prevOpIndex].num_outputs = slices;
                I8 **names = (I8 **)mt_new_storage(slices * sizeof(I8 *));

                for (U32 j = 0; j < slices - 1; j++) {
                    names[j] = spec->ops[prevOpIndex].output_tensors_name[j];
                }
                names[slices - 1] = spec->ops[curOpIndex].output_tensors_name[0];
                delete spec->ops[prevOpIndex].output_tensors_name;
                delete spec->ops[curOpIndex].output_tensors_name;
                spec->ops[curOpIndex].output_tensors_name = nullptr;
                spec->ops[curOpIndex].num_outputs = 0;
                spec->ops[prevOpIndex].output_tensors_name = names;

                // weight spec
                spec->ws[prevWeightIndex].bytes_of_weight = weightSize;
                spec->ws[prevWeightIndex].weight = weight;
                spec->ws[prevWeightIndex].bytes_of_vec = vecSize;
                spec->ws[prevWeightIndex].vec = vec;
                hasOptimized = true;

                setOperatorInvalid(spec, curOpIndex);
                i = curOpIndex;
            }
        }
        return hasOptimized;
    }
};
#endif
