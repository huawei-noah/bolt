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
#include "model_tools.h"
#include "model_print.h"
#include "model_optimizer.hpp"
#include <bitset>
#include "OPOptimizers/DeprecatedOPOptimizer.hpp"

F32 convert_F32(void* ptr, int index, DataType dt){
    F32 value = 0;
    switch (dt){
        case DT_F32: {
            value = ((F32*)ptr)[index];
            break;
        }
        case DT_I32: {
            value = ((I32*)ptr)[index];
            break;
        }
        case DT_U32: {
            value = ((U32*)ptr)[index];
            break;
        }
#ifdef _USE_FP16
        case DT_F16: {
            value = ((F16*)ptr)[index];
            break;
        }
        case DT_F16_8Q: {
            value = ((F16*)ptr)[index];
            break;
        }
#endif
        case DT_I8: {
            value = ((I8*)ptr)[index];
            break;
        }
        case DT_BIN01: {
            std::bitset<8> Val(((BIN8*)ptr)[index / 8]);
            if (Val.test(7 - (index % 8))) {
                value = 1.0;
            }
            break;
        }
        case DT_BIN11: {
            std::bitset<8> Val(((BIN8*)ptr)[index / 8]);
            if (Val.test(7 - (index % 8))) {
                value = 1.0;
            } else {
                value = -1.0;
            }
            break;
        }
        default:
            CHECK_REQUIREMENT(0);
            break;
    }
    return value;
}


void print_header(const ModelSpec ms){
    printf("[Model] %s\n", ms.model_name);
    printf("    [Input]");
    for(int i = 0; i < ms.num_inputs; i++){
        printf(" %s(", ms.input_names[i]);
        std::cout << tensorDesc2Str(ms.input_dims[i]);
        printf(")");
    }
    printf("\n");

    printf("    [Output]");
    for(int i = 0; i < ms.num_outputs; i++){
        printf(" %s", ms.output_names[i]);
    }
    printf("\n");
}


void print_operator_tensor_relationship(const ModelSpec ms, bool deleteDeprecatedOp){
    int number = ms.num_operator_specs;
    printf("    [Ops] %d\n", number);
    for(int i = 0; i < number; i++){
        if(deleteDeprecatedOp) {
            if(DeprecatedOPOptimizer::isDeprecatedOp(ms.ops[i].type))
                continue;
        }
#ifdef _DEBUG
        if (OT_Conv == ms.ops[i].type) {
            printf("Kernel shape is %d x %d\n", ms.ops[i].ps.conv_spec.kernel_size_h, ms.ops[i].ps.conv_spec.kernel_size_w);
        }
#endif
        printf("        Op %3d %32s %16s|", i, ms.ops[i].name, OperatorTypeName()[ms.ops[i].type]);
        for(U32 j = 0; j < ms.ops[i].num_inputs; j++){
            printf(" %s,", ms.ops[i].input_tensors_name[j]);
        }
        printf(" -> ");
        for(U32 j = 0; j < ms.ops[i].num_outputs; j++){
            printf(" %s,", ms.ops[i].output_tensors_name[j]);
        }
        printf("\n");
        if (nullptr != ms.ops[i].tensor_positions) {
            printf("        Tensor Positions: ");
            for (U32 j = 0; j < ms.ops[i].num_inputs + ms.ops[i].num_outputs; j++) {
                printf("%d ", ms.ops[i].tensor_positions[j]);
            }
            printf("\n");
        }
    }
}

void print_weights(const ModelSpec ms)
{
    int number = ms.num_weight_specs;
    printf("    [Weights] %d\n", number);
    for (int i = 0; i < number; i++) {
        if (DeprecatedOPOptimizer::isDeprecatedOpWeight(&ms, i)) {
            printf("        Weight %3d %32s | Delete mdt %d weight: %p %uB bias: %p %uB\n", i, ms.ws[i].op_name, ms.ws[i].mdt,
               ms.ws[i].weight, ms.ws[i].bytes_of_weight, ms.ws[i].vec, ms.ws[i].bytes_of_vec);
            continue;
        }

        printf("        Weight %3d %32s | Retain mdt %d weight: %p %uB bias: %p %uB example: ", i, ms.ws[i].op_name, ms.ws[i].mdt,
           ms.ws[i].weight, ms.ws[i].bytes_of_weight, ms.ws[i].vec, ms.ws[i].bytes_of_vec);
        if (ms.ws[i].bytes_of_weight > 0 && ms.ws[i].weight != nullptr) {
            printf("%f", convert_F32(ms.ws[i].weight, 0, ms.ws[i].mdt));
        }
        if (ms.ws[i].bytes_of_vec > 0 && ms.ws[i].vec != nullptr) {
            printf(",%f", convert_F32(ms.ws[i].vec, 0, ms.ws[i].mdt));
        }
        printf("\n");
    }
}


void print_relationship(const ModelSpec ms){
    int number = ms.num_op_tensor_entries;
    printf("    [Relationships] %d\n", number);
    for(int i = 0; i < number; i++){
        printf("        Relation %3d %32s |", i, ms.op_relationship_entries[i].op);
        for(U32 j = 0; j < ms.op_relationship_entries[i].num_inputs; j++){
            printf(" %s,", ms.op_relationship_entries[i].input_op_names[j]);
        }
        printf(" -> ");
        for(U32 j = 0; j < ms.op_relationship_entries[i].num_outputs; j++){
            printf(" %s,", ms.op_relationship_entries[i].output_op_names[j]);
        }
        printf("\n");
    }
}


void print_ms(const ModelSpec ms){
    print_header(ms);
    print_operator_tensor_relationship(ms);
    print_weights(ms);
    print_relationship(ms);
}
