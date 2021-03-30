// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "model_print.h"
#include "model_common.h"

void print_header(const ModelSpec ms)
{
    printf("[Model] %s\n    [DataType] %s\n    [Inputs] %d\n", ms.model_name, DataTypeName()[ms.dt],
        ms.num_inputs);
    if (ms.num_inputs > 0) {
        printf("        id name(data_descriptor)\n");
    }
    for (int i = 0; i < ms.num_inputs; i++) {
        printf(
            "        %2d %s(%s)\n", i, ms.input_names[i], tensorDesc2Str(ms.input_dims[i]).c_str());
    }
    printf("    [Outputs] %d\n", ms.num_outputs);
    if (ms.num_outputs > 0) {
        printf("        id name\n");
    }
    for (int i = 0; i < ms.num_outputs; i++) {
        printf("        %2d %s\n", i, ms.output_names[i]);
    }
}

void print_operator_tensor_relationship(const ModelSpec ms, bool deleteDeprecatedOp)
{
    int number = ms.num_operator_specs;
    printf("    [Operators] %d\n", number);
    if (number > 0) {
        printf("        %3s %32s %16s| inputs(reuse_positions) -> outputs(reuse_positions)    "
               "quantization_scale\n",
            "id", "name", "type");
    }
    for (int i = 0; i < number; i++) {
        if (deleteDeprecatedOp) {
            if (isDeprecatedOp(ms.ops[i].type)) {
                continue;
            }
        }
        printf("        %3d %32s %16s|", i, ms.ops[i].name, OperatorTypeName()[ms.ops[i].type]);
        for (U32 j = 0; j < ms.ops[i].num_inputs; j++) {
            int pos = -1;
            if (ms.ops[i].tensor_positions != nullptr) {
                pos = ms.ops[i].tensor_positions[j];
            }
            printf(" %s(%d)", ms.ops[i].input_tensors_name[j], pos);
        }
        printf(" ->");
        for (U32 j = 0; j < ms.ops[i].num_outputs; j++) {
            int pos = -1;
            if (ms.ops[i].tensor_positions != nullptr) {
                pos = ms.ops[i].tensor_positions[ms.ops[i].num_inputs + j];
            }
            printf(" %s(%d)", ms.ops[i].output_tensors_name[j], pos);
        }
        if (nullptr != ms.ops[i].feature_scale) {
            for (U32 j = 0; j < ms.ops[i].num_quant_feature; j++) {
                printf(" %f", ms.ops[i].feature_scale[j].scale[0]);
            }
        }
        printf("\n");
    }
}

void print_weights(const ModelSpec ms)
{
    int number = ms.num_weight_specs;
    printf("    [Weights] %d\n", number);
    if (number > 0) {
        printf("        %3s %32s | status %16s %12s %10s %14s %14s\n", "id", "name", "data_type",
            "weight_bytes", "bias_bytes", "weight_data", "bias_data");
    }
    for (int i = 0; i < number; i++) {
        if (isDeprecatedOpWeight(&ms, i)) {
            printf("        %3d %32s | delete %16s %12u %10u\n", i, ms.ws[i].op_name,
                DataTypeName()[ms.ws[i].mdt], ms.ws[i].bytes_of_weight, ms.ws[i].bytes_of_vec);
            continue;
        }

        printf("        %3d %32s | retain %16s %12u %10u", i, ms.ws[i].op_name,
            DataTypeName()[ms.ws[i].mdt], ms.ws[i].bytes_of_weight, ms.ws[i].bytes_of_vec);
        if (ms.ws[i].bytes_of_weight > 0 && ms.ws[i].weight != nullptr) {
            F32 value;
            transformToFloat(ms.ws[i].mdt, ms.ws[i].weight, &value, 1);
            printf(" %10.4f ...", value);
        } else if ((ms.ws[i].bytes_of_weight == 0 && ms.ws[i].weight != nullptr) ||
            (ms.ws[i].bytes_of_weight != 0 && ms.ws[i].weight == nullptr)) {
            UNI_ERROR_LOG("weight is null but size is not zero.\n");
        }
        if (ms.ws[i].bytes_of_vec > 0 && ms.ws[i].vec != nullptr) {
            DataType dt = ms.ws[i].mdt;
            if (DT_BIN01 == ms.ws[i].mdt || DT_BIN11 == ms.ws[i].mdt) {
                dt = DT_F16;
            }
            F32 value;
            transformToFloat(dt, ms.ws[i].vec, &value, 1);
            printf(" %10.4f ...", value);
        } else if ((ms.ws[i].bytes_of_vec == 0 && ms.ws[i].vec != nullptr) ||
            (ms.ws[i].bytes_of_vec != 0 && ms.ws[i].vec == nullptr)) {
            UNI_ERROR_LOG("bias is null but size is not zero.\n");
        }
        printf("\n");
    }
}

void print_relationship(const ModelSpec ms)
{
    int number = ms.num_op_tensor_entries;
    printf("    [Relationships] %d\n", number);
    if (number > 0) {
        printf("        %3s %32s | inputs -> outputs\n", "id", "name");
    }
    for (int i = 0; i < number; i++) {
        printf("        %3d %32s |", i, ms.op_relationship_entries[i].op);
        for (U32 j = 0; j < ms.op_relationship_entries[i].num_inputs; j++) {
            printf(" %s", ms.op_relationship_entries[i].input_op_names[j]);
        }
        printf(" ->");
        for (U32 j = 0; j < ms.op_relationship_entries[i].num_outputs; j++) {
            printf(" %s", ms.op_relationship_entries[i].output_op_names[j]);
        }
        printf("\n");
    }
}

void print_ms(const ModelSpec ms)
{
    print_header(ms);
    print_operator_tensor_relationship(ms);
    print_weights(ms);
    print_relationship(ms);
}
