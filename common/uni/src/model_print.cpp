// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <stdio.h>
#include "model_print.h"
#include "types.h"

void print_header(const ModelSpec ms)
{
    printf("[Model] %s\n    [Input]", ms.model_name);
    for (int i = 0; i < ms.num_inputs; i++) {
        printf(" %s(%s)", ms.input_names[i], tensorDesc2Str(ms.input_dims[i]).c_str());
    }
    printf("\n    [Output]");
    for (int i = 0; i < ms.num_outputs; i++) {
        printf(" %s", ms.output_names[i]);
    }
    printf("\n");
}

void print_operator_tensor_relationship(const ModelSpec ms, bool deleteDeprecatedOp)
{
    int number = ms.num_operator_specs;
    printf("    [Ops] %d\n", number);
    for (int i = 0; i < number; i++) {
        if (deleteDeprecatedOp) {
            if (isDeprecatedOp(ms.ops[i].type)) {
                continue;
            }
        }
        printf("        Op %3d %32s %16s|", i, ms.ops[i].name, OperatorTypeName()[ms.ops[i].type]);
        for (U32 j = 0; j < ms.ops[i].num_inputs; j++) {
            printf(" %s", ms.ops[i].input_tensors_name[j]);
        }
        printf(" ->");
        for (U32 j = 0; j < ms.ops[i].num_outputs; j++) {
            printf(" %s", ms.ops[i].output_tensors_name[j]);
        }
        if (nullptr != ms.ops[i].tensor_positions) {
            printf("    tensor position:");
            for (U32 j = 0; j < ms.ops[i].num_inputs + ms.ops[i].num_outputs; j++) {
                printf(" %d", ms.ops[i].tensor_positions[j]);
            }
        }
        if (nullptr != ms.ops[i].feature_scale) {
            printf("    quant scale:");
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
    for (int i = 0; i < number; i++) {
        if (isDeprecatedOpWeight(&ms, i)) {
            printf("        Weight %3d %32s | Delete mdt %d weight: %p %uB bias: %p %uB\n", i,
                ms.ws[i].op_name, ms.ws[i].mdt, ms.ws[i].weight, ms.ws[i].bytes_of_weight,
                ms.ws[i].vec, ms.ws[i].bytes_of_vec);
            continue;
        }

        printf("        Weight %3d %32s | Retain mdt %d weight: %p %uB bias: %p %uB example: ", i,
            ms.ws[i].op_name, ms.ws[i].mdt, ms.ws[i].weight, ms.ws[i].bytes_of_weight, ms.ws[i].vec,
            ms.ws[i].bytes_of_vec);
        if (ms.ws[i].bytes_of_weight > 0 && ms.ws[i].weight != nullptr) {
            F32 value;
            transformToFloat(ms.ws[i].mdt, ms.ws[i].weight, &value, 1);
            printf("%f", value);
        } else if ((ms.ws[i].bytes_of_weight == 0 && ms.ws[i].weight != nullptr) ||
            (ms.ws[i].bytes_of_weight != 0 && ms.ws[i].weight == nullptr)) {
            UNI_ERROR_LOG("weight is null but size is not zero\n");
        }
        if (ms.ws[i].bytes_of_vec > 0 && ms.ws[i].vec != nullptr) {
            DataType dt = ms.ws[i].mdt;
            if (DT_BIN01 == ms.ws[i].mdt || DT_BIN11 == ms.ws[i].mdt) {
                dt = DT_F16;
            }
            F32 value;
            transformToFloat(dt, ms.ws[i].vec, &value, 1);
            printf(",%f", value);
        } else if ((ms.ws[i].bytes_of_vec == 0 && ms.ws[i].vec != nullptr) ||
            (ms.ws[i].bytes_of_vec != 0 && ms.ws[i].vec == nullptr)) {
            UNI_ERROR_LOG("vec is null but size is not zero\n");
        }
        printf("\n");
    }
}

void print_relationship(const ModelSpec ms)
{
    int number = ms.num_op_tensor_entries;
    printf("    [Relationships] %d\n", number);
    for (int i = 0; i < number; i++) {
        printf("        Relation %3d %32s |", i, ms.op_relationship_entries[i].op);
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
